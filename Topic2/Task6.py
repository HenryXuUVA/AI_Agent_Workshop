# Task6.py
# LangGraph agent using Message API with switchable Llama/Qwen and
# participant-aware chat history remapping.

import argparse
from typing import Annotated, List, Literal, TypedDict
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

Speaker = Literal["Human", "Llama", "Qwen"]
ModelName = Literal["llama", "qwen"]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    should_exit: bool
    skip_llm: bool
    trace_enabled: bool
    active_model: str


def get_device() -> str:
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    print("Using CPU for inference")
    return "cpu"


def create_model_llm(model_id: str, device: str):
    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded/cached...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"Model loaded successfully: {model_id}")
    return llm


def _trace(state: AgentState, message: str) -> None:
    if state.get("trace_enabled", False):
        print(f"[TRACE] {message}")


def _model_label(model: ModelName) -> Speaker:
    return "Llama" if model == "llama" else "Qwen"


def _parse_prefixed_content(content: str) -> tuple[Speaker, str]:
    for speaker in ("Human", "Llama", "Qwen"):
        prefix = f"{speaker}:"
        if content.startswith(prefix):
            return speaker, content[len(prefix):].strip()
    # Fallback for malformed history.
    return "Human", content.strip()


def _system_prompt_for(target: ModelName) -> str:
    me = "Llama" if target == "llama" else "Qwen"
    other = "Qwen" if target == "llama" else "Llama"
    return (
        f"You are {me}. Participants in this conversation are Human, {me}, and {other}. "
        f"Conversation history may include prefixed names like 'Human:', '{me}:', and '{other}:'. "
        f"Respond as {me}, stay consistent with prior context, and keep responses concise and helpful."
    )


def _history_for_target(messages: List[BaseMessage], target: ModelName) -> List[BaseMessage]:
    """
    Convert canonical prefixed transcript into target-specific Message API roles.

    Rule:
    - target speaker messages -> assistant
    - all other speakers (Human + the other model) -> user
    """
    target_label = _model_label(target)
    out: List[BaseMessage] = [SystemMessage(content=_system_prompt_for(target))]

    for msg in messages:
        speaker, text = _parse_prefixed_content(msg.content)
        normalized = f"{speaker}: {text}"
        if speaker == target_label:
            out.append(AIMessage(content=normalized))
        else:
            out.append(HumanMessage(content=normalized))

    return out


def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "tool"
        lines.append(f"{role}: {msg.content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _invoke_model(llm, history: List[BaseMessage]) -> str:
    prompt = _messages_to_prompt(history)
    response = llm.invoke(prompt)
    return response if isinstance(response, str) else str(response)


def create_graph(llama_llm, qwen_llm):
    def get_user_input(state: AgentState) -> dict:
        _trace(
            state,
            "node=get_user_input entered "
            f"(should_exit={state.get('should_exit')}, skip_llm={state.get('skip_llm')})",
        )

        print("\n" + "=" * 60)
        print("Enter text ('Hey Qwen ...' routes to Qwen, 'verbose'/'quiet', or 'quit'):")
        print("=" * 60)
        print("\n> ", end="")

        user_input = input()
        normalized = user_input.strip().lower()

        if normalized in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"should_exit": True, "skip_llm": True}

        if normalized == "verbose":
            print("Tracing enabled.")
            return {"skip_llm": True, "trace_enabled": True}

        if normalized == "quiet":
            print("Tracing disabled.")
            return {"skip_llm": True, "trace_enabled": False}

        if normalized == "":
            print("Empty input ignored. Please enter some text.")
            return {"skip_llm": True}

        return {
            "skip_llm": False,
            "messages": [HumanMessage(content=f"Human: {user_input}")],
        }

    def call_llama(state: AgentState) -> dict:
        _trace(state, "node=call_llama entered")
        print("\nRunning Llama...")

        history = _history_for_target(state.get("messages", []), "llama")
        response = _invoke_model(llama_llm, history)

        _trace(state, f"node=call_llama completed (response_len={len(response)})")
        return {
            "active_model": "llama",
            "messages": [AIMessage(content=f"Llama: {response}")],
        }

    def call_qwen(state: AgentState) -> dict:
        _trace(state, "node=call_qwen entered")
        print("\nRunning Qwen...")

        history = _history_for_target(state.get("messages", []), "qwen")
        response = _invoke_model(qwen_llm, history)

        _trace(state, f"node=call_qwen completed (response_len={len(response)})")
        return {
            "active_model": "qwen",
            "messages": [AIMessage(content=f"Qwen: {response}")],
        }

    def print_response(state: AgentState) -> dict:
        _trace(
            state,
            "node=print_response entered "
            f"(active_model={state.get('active_model', '')})",
        )

        latest = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                latest = msg.content
                break

        print("\n" + "-" * 60)
        print("Model Response:")
        print("-" * 60)
        print(latest)
        return {}

    def route_after_input(state: AgentState) -> str:
        _trace(
            state,
            "route_after_input "
            f"(should_exit={state.get('should_exit')}, skip_llm={state.get('skip_llm')})",
        )

        if state.get("should_exit", False):
            return END

        if state.get("skip_llm", False):
            return "get_user_input"

        latest_human = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                speaker, text = _parse_prefixed_content(msg.content)
                if speaker == "Human":
                    latest_human = text
                    break

        if latest_human.strip().lower().startswith("hey qwen"):
            return "call_qwen"
        return "call_llama"

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            END: END,
        },
    )
    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")


class _ScriptedFakeLLM:
    def __init__(self, name: str):
        self.name = name

    def invoke(self, prompt: str) -> str:
        # Use the latest user line as intent for deterministic testing/demo.
        last_user = ""
        for line in prompt.splitlines()[::-1]:
            if line.startswith("user:"):
                last_user = line.split("user:", 1)[1].strip()
                break

        if self.name == "Llama":
            if "best ice cream flavor" in last_user.lower():
                return "There is no one best flavor, but vanilla is the most popular."
            if "i agree" in last_user.lower():
                return "Glad we agree. Taste is personal, but balance and texture matter too."
            return "From Llama: I think through tradeoffs before answering."

        if "what do you think" in last_user.lower():
            return "No way, chocolate is the best!"
        return "From Qwen: I prefer bold flavors and strong opinions."


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "should_exit": False,
        "skip_llm": False,
        "trace_enabled": False,
        "active_model": "",
    }


def _format_transcript(messages: List[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        lines.append(msg.content)
    return "\n".join(lines)


def run_self_test() -> None:
    # Validate history mapping examples from the assignment text.
    canonical = [
        HumanMessage(content="Human: What is the best ice cream flavor?"),
        AIMessage(content="Llama: There is no one best flavor, but the most popular is vanilla."),
    ]
    qwen_history = _history_for_target(canonical, "qwen")
    assert isinstance(qwen_history[1], HumanMessage)
    assert qwen_history[1].content == "Human: What is the best ice cream flavor?"
    assert isinstance(qwen_history[2], HumanMessage)
    assert qwen_history[2].content == "Llama: There is no one best flavor, but the most popular is vanilla."

    canonical.extend(
        [
            AIMessage(content="Qwen: No way, chocolate is the best!"),
            HumanMessage(content="Human: I agree."),
        ]
    )
    llama_history = _history_for_target(canonical, "llama")
    # After system: user, assistant, user, user
    assert isinstance(llama_history[1], HumanMessage)
    assert llama_history[1].content == "Human: What is the best ice cream flavor?"
    assert isinstance(llama_history[2], AIMessage)
    assert llama_history[2].content == "Llama: There is no one best flavor, but the most popular is vanilla."
    assert isinstance(llama_history[3], HumanMessage)
    assert llama_history[3].content == "Qwen: No way, chocolate is the best!"
    assert isinstance(llama_history[4], HumanMessage)
    assert llama_history[4].content == "Human: I agree."

    # Graph routing/invocation smoke test with fake models.
    graph = create_graph(_ScriptedFakeLLM("Llama"), _ScriptedFakeLLM("Qwen"))
    with patch(
        "builtins.input",
        side_effect=[
            "What is the best ice cream flavor?",
            "Hey Qwen, what do you think?",
            "I agree.",
            "quit",
        ],
    ):
        out = graph.invoke(_initial_state())

    all_text = [m.content for m in out.get("messages", [])]
    assert any(t.startswith("Llama:") for t in all_text)
    assert any(t.startswith("Qwen:") for t in all_text)
    print("self_test_ok")


def record_interesting_conversations(output_path: str) -> None:
    graph = create_graph(_ScriptedFakeLLM("Llama"), _ScriptedFakeLLM("Qwen"))

    conversations = [
        [
            "What is the best ice cream flavor?",
            "Hey Qwen, what do you think?",
            "I agree.",
            "quit",
        ],
        [
            "Design a tiny weekend trip plan.",
            "Hey Qwen, make it more adventurous.",
            "Now make it cheaper.",
            "quit",
        ],
    ]

    sections: List[str] = []
    for idx, script in enumerate(conversations, start=1):
        with patch("builtins.input", side_effect=script):
            out = graph.invoke(_initial_state())

        transcript = _format_transcript(out.get("messages", []))
        sections.append(f"Conversation {idx}\n{'=' * 40}\n{transcript}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"recorded_conversations_written: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run local tests without loading HF models")
    parser.add_argument(
        "--record-demo",
        action="store_true",
        help="Record scripted interesting conversations using fake models",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.record_demo:
        record_interesting_conversations("d:\\AgenticAI\\Topic2\\task6_interesting_conversations.txt")
        return

    print("=" * 60)
    print("LangGraph Message API Agent (Llama + Qwen with shared history)")
    print("=" * 60)

    device = get_device()
    llama_llm = create_model_llm("meta-llama/Llama-3.2-1B-Instruct", device)
    qwen_llm = create_model_llm("Qwen/Qwen2.5-0.5B-Instruct", device)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    graph.invoke(_initial_state())


if __name__ == "__main__":
    main()
