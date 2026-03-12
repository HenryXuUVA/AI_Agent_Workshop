import argparse
import json
import os
import pickle
import re
import threading
import time
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI

from Task4 import OPENAI_TOOLS, TOOL_REGISTRY, execute_tool_call

try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINTER_MODE = "sqlite"
except ImportError:
    SqliteSaver = None
    CHECKPOINTER_MODE = "memory"

from langgraph.checkpoint.memory import InMemorySaver


SYSTEM_PROMPT = (
    "You are a helpful assistant in an ongoing conversation. Use tools for weather, "
    "geometry calculations, letter counts, and total character counts when they are relevant. "
    "Use prior conversation context when the user refers to earlier turns."
)

SUPPORTED_LOCATIONS = ("San Francisco", "New York", "London", "Tokyo")


class FileCheckpointSaver(InMemorySaver):
    """Disk-backed fallback checkpointer when sqlite is unavailable."""

    _global_lock = threading.Lock()

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        with self._global_lock:
            if not os.path.exists(self.file_path):
                return
            with open(self.file_path, "rb") as file_handle:
                payload = pickle.load(file_handle)

        loaded_storage = payload.get("storage", {})
        self.storage = defaultdict(lambda: defaultdict(dict))
        for thread_id, namespace_map in loaded_storage.items():
            self.storage[thread_id] = defaultdict(dict, namespace_map)
        self.writes = defaultdict(dict, payload.get("writes", {}))
        self.blobs = payload.get("blobs", {})

    def _plain_storage(self) -> dict[str, Any]:
        plain: dict[str, Any] = {}
        for thread_id, namespace_map in self.storage.items():
            plain[thread_id] = dict(namespace_map)
        return plain

    def _persist_to_disk(self) -> None:
        with self._global_lock:
            os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
            for attempt in range(5):
                try:
                    with open(self.file_path, "wb") as file_handle:
                        pickle.dump(
                            {
                                "storage": self._plain_storage(),
                                "writes": dict(self.writes),
                                "blobs": dict(self.blobs),
                            },
                            file_handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05)

    def put(self, config, checkpoint, metadata, new_versions):
        output_config = super().put(config, checkpoint, metadata, new_versions)
        self._persist_to_disk()
        return output_config

    def put_writes(self, config, writes, task_id, task_path=""):
        super().put_writes(config, writes, task_id, task_path)
        self._persist_to_disk()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    should_exit: bool
    skip_turn: bool


def _to_openai_message(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["args"]),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return payload
    raise TypeError(f"Unsupported message type: {type(message)!r}")


class OpenAIToolModel:
    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[_to_openai_message(message) for message in messages],
            tools=OPENAI_TOOLS,
        )

        assistant_message = response.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "args": json.loads(tool_call.function.arguments),
                        "id": tool_call.id,
                        "type": "tool_call",
                    }
                )

        return AIMessage(content=assistant_message.content or "", tool_calls=tool_calls)


class ScriptedConversationModel:
    def _find_last_location(self, messages: list[BaseMessage]) -> str | None:
        for message in reversed(messages):
            content = getattr(message, "content", "")
            for location in SUPPORTED_LOCATIONS:
                if location.lower() in content.lower():
                    return location
        return None

    def _has_circle_context(self, messages: list[BaseMessage]) -> bool:
        for message in reversed(messages):
            content = getattr(message, "content", "")
            if "circle" in content.lower() or "circle_area" in content.lower():
                return True
        return False

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        last_message = messages[-1]

        if isinstance(last_message, HumanMessage):
            text = last_message.content.strip()
            lowered = text.lower()

            if "weather" in lowered:
                for location in SUPPORTED_LOCATIONS:
                    if location.lower() in lowered:
                        return AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "get_weather",
                                    "args": {"location": location},
                                    "id": "tool_weather",
                                    "type": "tool_call",
                                }
                            ],
                        )

            if "that city name" in lowered:
                location = self._find_last_location(messages[:-1])
                if location:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "count_characters",
                                "args": {"text": location, "include_spaces": True},
                                "id": "tool_count_city_name",
                                "type": "tool_call",
                            }
                        ],
                    )
                return AIMessage(content="I do not have a city in context yet.")

            if "how many" in lowered and "mississippi riverboats" in lowered:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "count_letter_occurrences",
                            "args": {
                                "text": "Mississippi riverboats",
                                "letter": "s",
                                "case_sensitive": False,
                            },
                            "id": "tool_letter_count",
                            "type": "tool_call",
                        }
                    ],
                )

            radius_match = re.search(r"radius\s+(\d+(?:\.\d+)?)", lowered)
            if "circle" in lowered and "area" in lowered and radius_match:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "geometry_calculator",
                            "args": {
                                "operation": "circle_area",
                                "radius": float(radius_match.group(1)),
                            },
                            "id": "tool_circle_area",
                            "type": "tool_call",
                        }
                    ],
                )

            if "what about with radius" in lowered and radius_match and self._has_circle_context(messages[:-1]):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "geometry_calculator",
                            "args": {
                                "operation": "circle_area",
                                "radius": float(radius_match.group(1)),
                            },
                            "id": "tool_circle_area_followup",
                            "type": "tool_call",
                        }
                    ],
                )

            if "how many characters" in lowered:
                text_match = re.search(
                    r"(?:characters\s+are\s+in|characters\s+in)\s+(.+?)[?.!]?$",
                    text,
                    re.IGNORECASE,
                )
                count_target = text_match.group(1).strip() if text_match else text
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "count_characters",
                            "args": {"text": count_target, "include_spaces": True},
                            "id": "tool_character_count",
                            "type": "tool_call",
                        }
                    ],
                )

            return AIMessage(
                content="I can help with weather, geometry, letter counts, and character counts."
            )

        if isinstance(last_message, ToolMessage):
            payload = json.loads(last_message.content)
            if not payload.get("success", False):
                return AIMessage(content=f"Tool error: {payload['error']}")

            if "forecast" in payload:
                return AIMessage(
                    content=f"The weather in {payload['location']} is {payload['forecast']}."
                )
            if "letter" in payload and "count" in payload:
                return AIMessage(
                    content=(
                        f"There are {payload['count']} occurrences of '{payload['letter']}' "
                        f'in "{payload["text"]}".'
                    )
                )
            if payload.get("operation") == "circle_area":
                return AIMessage(
                    content=f"The area of the circle is {payload['result']}."
                )
            if "include_spaces" in payload and "count" in payload:
                return AIMessage(
                    content=f'The text "{payload["text"]}" contains {payload["count"]} characters.'
                )

        return AIMessage(content="I need a user question to continue.")


def create_graph(model, tool_registry: dict[str, Any], checkpointer):
    def get_user_input(state: AgentState) -> dict[str, Any]:
        print("\n" + "=" * 60)
        print("Enter your text ('quit' to exit):")
        print("=" * 60)
        print("\n> ", end="")

        user_input = input().strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return {"should_exit": True, "skip_turn": True}
        if not user_input:
            print("Empty input ignored.")
            return {"should_exit": False, "skip_turn": True}

        return {
            "should_exit": False,
            "skip_turn": False,
            "messages": [HumanMessage(content=user_input)],
        }

    def call_model(state: AgentState) -> dict[str, Any]:
        print("\nCalling model...")
        return {"messages": [model.invoke(state["messages"])]}

    def run_tools(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {}

        tool_messages: list[ToolMessage] = []
        for tool_call in last_message.tool_calls:
            tool_message, tool_log = execute_tool_call(tool_call, tool_registry)
            print(tool_log)
            tool_messages.append(tool_message)

        return {"messages": tool_messages}

    def print_response(state: AgentState) -> dict[str, Any]:
        last_ai = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage) and message.content),
            None,
        )
        if last_ai is not None:
            print("\n" + "-" * 60)
            print("Assistant:")
            print("-" * 60)
            print(last_ai.content)
        return {}

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if state.get("skip_turn", False):
            return "get_user_input"
        return "call_model"

    def route_after_model(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "run_tools"
        return "print_response"

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("run_tools", run_tools)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_model": "call_model",
            END: END,
        },
    )
    graph_builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "run_tools": "run_tools",
            "print_response": "print_response",
        },
    )
    graph_builder.add_edge("run_tools", "call_model")
    graph_builder.add_edge("print_response", "get_user_input")
    return graph_builder.compile(checkpointer=checkpointer)


def initial_state() -> AgentState:
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "should_exit": False,
        "skip_turn": False,
    }


def open_checkpointer(checkpoint_path: Path):
    if CHECKPOINTER_MODE == "sqlite" and SqliteSaver is not None:
        return SqliteSaver.from_conn_string(str(checkpoint_path))
    return nullcontext(FileCheckpointSaver(str(checkpoint_path)))


def resume_or_start(graph, config: dict[str, Any], starting_state: AgentState):
    snapshot = graph.get_state(config)
    has_messages = bool(snapshot and snapshot.values and snapshot.values.get("messages"))
    if has_messages:
        values = snapshot.values or {}
        if values.get("should_exit", False):
            print(
                f"Recovered checkpoint for thread_id={config['configurable']['thread_id']}; "
                "continuing with saved conversation context..."
            )
            return graph.invoke({"should_exit": False, "skip_turn": False}, config=config)

        print(f"Resuming in-progress conversation for thread_id={config['configurable']['thread_id']}...")
        return graph.invoke(None, config=config)

    print(f"Starting new conversation for thread_id={config['configurable']['thread_id']}...")
    return graph.invoke(starting_state, config=config)


def save_mermaid_diagram(graph, output_path: Path) -> str:
    mermaid = graph.get_graph(xray=True).draw_mermaid()
    output_path.write_text(mermaid, encoding="utf-8")
    return mermaid


def format_messages(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            lines.append(f"System: {message.content}")
        elif isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    lines.append(f"Assistant -> Tool {tool_call['name']}: {tool_call['args']}")
            elif message.content:
                lines.append(f"Assistant: {message.content}")
        elif isinstance(message, ToolMessage):
            lines.append(f"Tool Result: {message.content}")
    return "\n".join(lines)


def run_scripted_session(graph, config: dict[str, Any], scripted_inputs: list[str], starting_state: AgentState) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        with patch("builtins.input", side_effect=scripted_inputs):
            resume_or_start(graph, config, starting_state)
    return buffer.getvalue()


def record_portfolio_artifacts() -> None:
    base_dir = Path(__file__).resolve().parent
    checkpoint_path = base_dir / "task5_demo_checkpoints.bin"
    mermaid_path = base_dir / "task5_graph.mmd"
    portfolio_path = base_dir / "task5_portfolio.md"
    thread_id = "portfolio-demo"

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    first_inputs = [
        "What's the weather like in San Francisco?",
        "quit",
    ]
    second_inputs = [
        "How many characters are in that city name?",
        "How many s are in Mississippi riverboats?",
        "What is the area of a circle with radius 3?",
        "What about with radius 4 instead?",
        "quit",
    ]

    with open_checkpointer(checkpoint_path) as checkpointer:
        graph = create_graph(ScriptedConversationModel(), TOOL_REGISTRY, checkpointer)
        mermaid = save_mermaid_diagram(graph, mermaid_path)
        config = {"configurable": {"thread_id": thread_id}}

        session_one_log = run_scripted_session(graph, config, first_inputs, initial_state())
        session_one_snapshot = graph.get_state(config).values or {}
        session_one_messages = list(session_one_snapshot.get("messages", []))

    with open_checkpointer(checkpoint_path) as checkpointer:
        graph = create_graph(ScriptedConversationModel(), TOOL_REGISTRY, checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        session_two_log = run_scripted_session(graph, config, second_inputs, initial_state())
        session_two_snapshot = graph.get_state(config).values or {}
        session_two_messages = list(session_two_snapshot.get("messages", []))

    portfolio_text = "\n".join(
        [
            "# Task 5 Portfolio",
            "",
            "## Mermaid Diagram",
            "```mermaid",
            mermaid.rstrip(),
            "```",
            "",
            "## Session 1 Before Recovery",
            "```text",
            session_one_log.rstrip(),
            "```",
            "",
            "## Session 1 Transcript",
            "```text",
            format_messages(session_one_messages),
            "```",
            "",
            "## Session 2 After Recovery",
            "```text",
            session_two_log.rstrip(),
            "```",
            "",
            "## Full Transcript After Recovery",
            "```text",
            format_messages(session_two_messages),
            "```",
            "",
            "## Recovery Notes",
            "- Session 2 reused the same thread id and checkpoint file as Session 1.",
            "- The recovered conversation answered \"How many characters are in that city name?\" by using the saved San Francisco context from Session 1.",
            f"- Checkpoint file: {checkpoint_path.name}",
            f"- Mermaid source file: {mermaid_path.name}",
            "",
        ]
    )
    portfolio_path.write_text(portfolio_text, encoding="utf-8")
    print(f"portfolio_written: {portfolio_path}")
    print(f"mermaid_written: {mermaid_path}")


def run_self_test() -> None:
    base_dir = Path(__file__).resolve().parent
    checkpoint_path = base_dir / "task5_test_checkpoints.bin"
    mermaid_path = base_dir / "task5_graph.mmd"
    portfolio_path = base_dir / "task5_portfolio.md"
    config = {"configurable": {"thread_id": "self-test"}}

    for path in (checkpoint_path, mermaid_path, portfolio_path):
        if path.exists() and path.suffix in {".bin", ".mmd", ".md"}:
            path.unlink()

    with open_checkpointer(checkpoint_path) as checkpointer:
        graph = create_graph(ScriptedConversationModel(), TOOL_REGISTRY, checkpointer)
        mermaid = save_mermaid_diagram(graph, mermaid_path)
        assert "get_user_input" in mermaid

        run_scripted_session(
            graph,
            config,
            ["What's the weather like in San Francisco?", "quit"],
            initial_state(),
        )
        snapshot_one = graph.get_state(config).values or {}
        messages_one = list(snapshot_one.get("messages", []))
        assert any("San Francisco" in getattr(message, "content", "") for message in messages_one)

    with open_checkpointer(checkpoint_path) as checkpointer:
        graph = create_graph(ScriptedConversationModel(), TOOL_REGISTRY, checkpointer)
        run_scripted_session(
            graph,
            config,
            [
                "How many characters are in that city name?",
                "What is the area of a circle with radius 3?",
                "What about with radius 4 instead?",
                "quit",
            ],
            initial_state(),
        )
        snapshot_two = graph.get_state(config).values or {}
        messages_two = list(snapshot_two.get("messages", []))

    all_contents = [getattr(message, "content", "") for message in messages_two]
    all_tool_calls = [
        tool_call
        for message in messages_two
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    assert any("San Francisco" in content and "13" in content for content in all_contents)
    assert any(
        tool_call["name"] == "geometry_calculator" and tool_call["args"].get("radius") == 4.0
        for tool_call in all_tool_calls
    )
    assert any("50.265" in content for content in all_contents)

    record_portfolio_artifacts()
    assert mermaid_path.exists()
    assert portfolio_path.exists()
    saved_portfolio = portfolio_path.read_text(encoding="utf-8")
    assert "How many characters are in that city name?" in saved_portfolio
    assert "Recovered checkpoint" in saved_portfolio
    print("self_test_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run local verification without OpenAI calls")
    parser.add_argument("--record-demo", action="store_true", help="Write portfolio artifacts using the scripted model")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use in interactive mode")
    parser.add_argument("--thread-id", default="task5-session", help="Checkpoint thread id")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.record_demo:
        record_portfolio_artifacts()
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Use --self-test, --record-demo, or configure the API key.")

    base_dir = Path(__file__).resolve().parent
    checkpoint_path = base_dir / ("task5_checkpoints.db" if CHECKPOINTER_MODE == "sqlite" else "task5_checkpoints.bin")

    with open_checkpointer(checkpoint_path) as checkpointer:
        graph = create_graph(OpenAIToolModel(args.model), TOOL_REGISTRY, checkpointer)
        save_mermaid_diagram(graph, base_dir / "task5_graph.mmd")
        resume_or_start(
            graph,
            {"configurable": {"thread_id": args.thread_id}},
            initial_state(),
        )


if __name__ == "__main__":
    main()
