"""
Tool-calling agent that uses LangChain-style tools with a manual loop.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI

from Task3 import geometry_calculator as task3_geometry_calculator


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools for weather, calculations, counting letters, "
    "and counting total characters whenever they are relevant."
)


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a supported location."""
    weather_data = {
        "San Francisco": "Sunny, 72 F",
        "New York": "Cloudy, 55 F",
        "London": "Rainy, 48 F",
        "Tokyo": "Clear, 65 F",
    }
    forecast = weather_data.get(location)
    if forecast is None:
        return json.dumps({"success": False, "error": f"Weather data not available for {location}"})
    return json.dumps({"success": True, "location": location, "forecast": forecast})


@tool
def geometry_calculator(
    operation: str,
    expression: str | None = None,
    radius: float | None = None,
    length: float | None = None,
    width: float | None = None,
    base: float | None = None,
    height: float | None = None,
    x1: float | None = None,
    y1: float | None = None,
    x2: float | None = None,
    y2: float | None = None,
) -> str:
    """Evaluate arithmetic expressions and geometry formulas."""
    payload = {"operation": operation}
    optional_fields = {
        "expression": expression,
        "radius": radius,
        "length": length,
        "width": width,
        "base": base,
        "height": height,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value
    return task3_geometry_calculator(json.dumps(payload))


@tool
def count_letter_occurrences(text: str, letter: str, case_sensitive: bool = False) -> str:
    """Count how many times one letter appears in a piece of text."""
    if len(letter) != 1:
        return json.dumps({"success": False, "error": "letter must be a single character"})

    haystack = text if case_sensitive else text.lower()
    needle = letter if case_sensitive else letter.lower()
    count = sum(1 for character in haystack if character == needle)
    return json.dumps(
        {
            "success": True,
            "text": text,
            "letter": letter,
            "case_sensitive": case_sensitive,
            "count": count,
        }
    )


@tool
def count_characters(text: str, include_spaces: bool = True) -> str:
    """Count the total number of characters in a piece of text."""
    normalized_text = text if include_spaces else text.replace(" ", "")
    return json.dumps(
        {
            "success": True,
            "text": text,
            "include_spaces": include_spaces,
            "count": len(normalized_text),
        }
    )


TOOLS = [get_weather, geometry_calculator, count_letter_occurrences, count_characters]
TOOL_REGISTRY = {tool_item.name: tool_item for tool_item in TOOLS}
OPENAI_TOOLS = [convert_to_openai_tool(tool_item) for tool_item in TOOLS]


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
        tool_calls = []
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


class ScriptedDemoModel:
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        last_message = messages[-1]

        if isinstance(last_message, HumanMessage):
            text = last_message.content.strip()
            lowered = text.lower()

            if "weather" in lowered:
                for location in ("San Francisco", "New York", "London", "Tokyo"):
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

            if "total number of characters" in lowered or "how many characters" in lowered:
                text_match = re.search(r"(?:characters\s+are\s+in|characters\s+in)\s+(.+?)[?.!]?$", text, re.IGNORECASE)
                count_target = text_match.group(1).strip() if text_match else text
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "count_characters",
                            "args": {
                                "text": count_target,
                                "include_spaces": True,
                            },
                            "id": "tool_character_count",
                            "type": "tool_call",
                        }
                    ],
                )

            return AIMessage(content="I can help with weather, calculations, letter counts, or EST time.")

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
                        f"in \"{payload['text']}\"."
                    )
                )
            if "operation" in payload:
                return AIMessage(
                    content=f"The {payload['operation']} result is {payload['result']}."
                )
            if "include_spaces" in payload and "count" in payload:
                return AIMessage(
                    content=(
                        f'The text "{payload["text"]}" contains {payload["count"]} '
                        f"characters."
                    )
                )

        return AIMessage(content="I need a user question to continue.")


def execute_tool_call(tool_call: dict[str, Any], tool_registry: dict[str, Any]) -> tuple[ToolMessage, str]:
    tool_name = tool_call["name"]
    tool_impl = tool_registry.get(tool_name)
    if tool_impl is None:
        result = json.dumps({"success": False, "error": f"Unknown tool: {tool_name}"})
    else:
        result = tool_impl.invoke(tool_call["args"])

    tool_message = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
    log_text = f"  Tool: {tool_name}\n  Args: {tool_call['args']}\n  Result: {result}"
    return tool_message, log_text


def run_agent(
    user_query: str,
    model,
    tool_registry: dict[str, Any] | None = None,
    max_iterations: int = 5,
    stream_to_stdout: bool = True,
) -> tuple[str, str]:
    registry = tool_registry or TOOL_REGISTRY
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]
    transcript: list[str] = [f"User: {user_query}", ""]

    def log(line: str = "") -> None:
        transcript.append(line)
        if stream_to_stdout:
            print(line)

    if stream_to_stdout:
        print(f"User: {user_query}\n")

    for iteration in range(1, max_iterations + 1):
        log(f"--- Iteration {iteration} ---")
        response = model.invoke(messages)

        if response.tool_calls:
            log(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            messages.append(response)

            for tool_call in response.tool_calls:
                tool_message, tool_log = execute_tool_call(tool_call, registry)
                log(tool_log)
                messages.append(tool_message)

            log()
            continue

        messages.append(response)
        log(f"Assistant: {response.content}")
        log()
        return response.content, "\n".join(transcript)

    final_text = "Max iterations reached"
    log(f"Assistant: {final_text}")
    return final_text, "\n".join(transcript)


def record_portfolio_runs(output_path: Path) -> None:
    prompts = [
        "What's the weather like in San Francisco?",
        "How many s are in Mississippi riverboats?",
        "What is the area of a circle with radius 3?",
        "How many characters are in OpenAI builds tools?",
    ]

    sections: list[str] = []
    model = ScriptedDemoModel()
    for index, prompt in enumerate(prompts, start=1):
        _, transcript = run_agent(prompt, model, stream_to_stdout=False)
        sections.append(f"Conversation {index}\n{'=' * 50}\n{transcript}")

    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"portfolio_output_written: {output_path}")


def run_self_test() -> None:
    weather_payload = json.loads(get_weather.invoke({"location": "San Francisco"}))
    assert weather_payload == {
        "success": True,
        "location": "San Francisco",
        "forecast": "Sunny, 72 F",
    }

    count_payload = json.loads(
        count_letter_occurrences.invoke(
            {"text": "Mississippi riverboats", "letter": "s", "case_sensitive": False}
        )
    )
    assert count_payload["count"] == 5

    calc_payload = json.loads(
        geometry_calculator.invoke({"operation": "evaluate", "expression": "(2 + 3) * 4"})
    )
    assert calc_payload == {"success": True, "operation": "evaluate", "result": 20.0}

    char_payload = json.loads(
        count_characters.invoke({"text": "OpenAI builds tools", "include_spaces": True})
    )
    assert char_payload == {
        "success": True,
        "text": "OpenAI builds tools",
        "include_spaces": True,
        "count": 19,
    }

    prompts = [
        "What's the weather like in San Francisco?",
        "How many s are in Mississippi riverboats?",
        "What is the area of a circle with radius 3?",
        "How many characters are in OpenAI builds tools?",
    ]
    model = ScriptedDemoModel()
    for prompt in prompts:
        final_text, _ = run_agent(prompt, model, stream_to_stdout=False)
        assert final_text

    portfolio_path = Path(__file__).with_name("task4_portfolio_output.txt")
    record_portfolio_runs(portfolio_path)
    assert portfolio_path.exists()
    saved_text = portfolio_path.read_text(encoding="utf-8")
    assert "Mississippi riverboats" in saved_text
    assert "circle_area" in saved_text
    print("self_test_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run local checks without OpenAI API calls")
    parser.add_argument(
        "--record-demo",
        action="store_true",
        help="Record scripted tool-calling transcripts for portfolio output",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name to use for live runs",
    )
    parser.add_argument("query", nargs="*", help="Optional one-shot user query")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.record_demo:
        portfolio_path = Path(__file__).with_name("task4_portfolio_output.txt")
        record_portfolio_runs(portfolio_path)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Use --self-test, --record-demo, or configure the API key."
        )

    model = OpenAIToolModel(args.model)
    if args.query:
        run_agent(" ".join(args.query), model)
        return

    print("=" * 60)
    print("Task4 Multi-Tool Agent")
    print("=" * 60)
    print("Enter 'quit' to exit.\n")

    while True:
        user_query = input("> ").strip()
        if user_query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_query:
            continue
        run_agent(user_query, model)


if __name__ == "__main__":
    main()
