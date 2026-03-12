import argparse
import ast
import json
import math
import os
import re
from typing import Annotated, Any, TypedDict
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Use the geometry_calculator tool for any "
    "arithmetic or geometry calculation instead of doing the math in your head. "
    "Supported operations are evaluate, circle_area, circle_circumference, "
    "rectangle_area, rectangle_perimeter, triangle_area, and distance_2d."
)

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "geometry_calculator",
        "description": (
            "Evaluate arithmetic expressions and geometry formulas. "
            "Pass arguments as JSON with an operation field and the required numeric inputs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "evaluate",
                        "circle_area",
                        "circle_circumference",
                        "rectangle_area",
                        "rectangle_perimeter",
                        "triangle_area",
                        "distance_2d",
                    ],
                },
                "expression": {"type": "string"},
                "radius": {"type": "number"},
                "length": {"type": "number"},
                "width": {"type": "number"},
                "base": {"type": "number"},
                "height": {"type": "number"},
                "x1": {"type": "number"},
                "y1": {"type": "number"},
                "x2": {"type": "number"},
                "y2": {"type": "number"},
            },
            "required": ["operation"],
        },
    },
}

ALLOWED_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}

ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    should_exit: bool


def _safe_eval(expression: str) -> float:
    node = ast.parse(expression, mode="eval")

    def _evaluate(current: ast.AST) -> float:
        if isinstance(current, ast.Expression):
            return _evaluate(current.body)
        if isinstance(current, ast.Constant) and isinstance(current.value, (int, float)):
            return float(current.value)
        if isinstance(current, ast.BinOp) and type(current.op) in ALLOWED_BINARY_OPS:
            left = _evaluate(current.left)
            right = _evaluate(current.right)
            return float(ALLOWED_BINARY_OPS[type(current.op)](left, right))
        if isinstance(current, ast.UnaryOp) and type(current.op) in ALLOWED_UNARY_OPS:
            operand = _evaluate(current.operand)
            return float(ALLOWED_UNARY_OPS[type(current.op)](operand))
        raise ValueError(f"Unsupported expression: {expression}")

    return _evaluate(node)


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number")
    return float(value)


def geometry_calculator(raw_input: str) -> str:
    try:
        payload = json.loads(raw_input)
        operation = payload["operation"]

        if operation == "evaluate":
            expression = payload.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError("'expression' must be a non-empty string")
            result = _safe_eval(expression)
        elif operation == "circle_area":
            radius = _require_number(payload, "radius")
            result = math.pi * radius * radius
        elif operation == "circle_circumference":
            radius = _require_number(payload, "radius")
            result = 2 * math.pi * radius
        elif operation == "rectangle_area":
            length = _require_number(payload, "length")
            width = _require_number(payload, "width")
            result = length * width
        elif operation == "rectangle_perimeter":
            length = _require_number(payload, "length")
            width = _require_number(payload, "width")
            result = 2 * (length + width)
        elif operation == "triangle_area":
            base = _require_number(payload, "base")
            height = _require_number(payload, "height")
            result = 0.5 * base * height
        elif operation == "distance_2d":
            x1 = _require_number(payload, "x1")
            y1 = _require_number(payload, "y1")
            x2 = _require_number(payload, "x2")
            y2 = _require_number(payload, "y2")
            result = math.dist((x1, y1), (x2, y2))
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return json.dumps(
            {
                "success": True,
                "operation": operation,
                "result": result,
            }
        )
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


def _tool_calls_from(message: AIMessage) -> list[dict[str, Any]]:
    return list(message.additional_kwargs.get("tool_calls", []))


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
        tool_calls = _tool_calls_from(message)
        if tool_calls:
            payload["tool_calls"] = tool_calls
        return payload
    raise TypeError(f"Unsupported message type: {type(message)!r}")


class OpenAIModel:
    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[_to_openai_message(message) for message in messages],
            tools=[CALCULATOR_TOOL],
        )

        assistant_message = response.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )

        kwargs = {"tool_calls": tool_calls} if tool_calls else {}
        return AIMessage(content=assistant_message.content or "", additional_kwargs=kwargs)


class FakeToolCallingModel:
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        last_message = messages[-1]

        if isinstance(last_message, HumanMessage):
            text = last_message.content.lower()

            radius_match = re.search(r"radius\s+(\d+(?:\.\d+)?)", text)
            if "circle" in text and "area" in text and radius_match:
                radius = float(radius_match.group(1))
                return AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_circle_area",
                                "type": "function",
                                "function": {
                                    "name": "geometry_calculator",
                                    "arguments": json.dumps(
                                        {"operation": "circle_area", "radius": radius}
                                    ),
                                },
                            }
                        ]
                    },
                )

            expression_match = re.search(r"evaluate\s+(.+?)[?.!]?$", last_message.content, re.IGNORECASE)
            if expression_match:
                expression = expression_match.group(1).strip()
                return AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_evaluate",
                                "type": "function",
                                "function": {
                                    "name": "geometry_calculator",
                                    "arguments": json.dumps(
                                        {"operation": "evaluate", "expression": expression}
                                    ),
                                },
                            }
                        ]
                    },
                )

            return AIMessage(content="Ask me for an arithmetic or geometry calculation.")

        if isinstance(last_message, ToolMessage):
            tool_result = json.loads(last_message.content)
            if not tool_result.get("success"):
                return AIMessage(content=f"Tool error: {tool_result['error']}")

            operation = tool_result["operation"]
            result = tool_result["result"]
            if operation == "circle_area":
                return AIMessage(content=f"The area of the circle is {result:.4f}.")
            if operation == "evaluate":
                return AIMessage(content=f"The expression evaluates to {result}.")
            return AIMessage(content=f"The result is {result}.")

        return AIMessage(content="I need a user request to continue.")


def create_graph(model) -> Any:
    def get_user_input(state: AgentState) -> dict[str, Any]:
        print("\n" + "=" * 60)
        print("Enter a math question ('quit' to exit):")
        print("=" * 60)
        print("\n> ", end="")

        user_input = input().strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return {"should_exit": True}
        if not user_input:
            print("Empty input ignored.")
            return {}
        return {
            "should_exit": False,
            "messages": [HumanMessage(content=user_input)],
        }

    def call_model(state: AgentState) -> dict[str, Any]:
        print("\nCalling model...")
        return {"messages": [model.invoke(state["messages"])]}

    def run_tool(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {}

        tool_messages: list[ToolMessage] = []
        for tool_call in _tool_calls_from(last_message):
            function_name = tool_call["function"]["name"]
            if function_name != "geometry_calculator":
                result = json.dumps({"success": False, "error": f"Unknown tool: {function_name}"})
            else:
                result = geometry_calculator(tool_call["function"]["arguments"])

            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_messages}

    def print_response(state: AgentState) -> dict[str, Any]:
        last_ai = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
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
        if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
            return "call_model"
        return "get_user_input"

    def route_after_model(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and _tool_calls_from(last_message):
            return "run_tool"
        return "print_response"

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("run_tool", run_tool)
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
            "run_tool": "run_tool",
            "print_response": "print_response",
        },
    )
    graph_builder.add_edge("run_tool", "call_model")
    graph_builder.add_edge("print_response", "get_user_input")
    return graph_builder.compile()


def save_graph_image(graph, filename: str = "lg_graph.png") -> None:
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as file_handle:
            file_handle.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as exc:
        print(f"Could not save graph image: {exc}")


def run_self_test() -> None:
    assert json.loads(geometry_calculator(json.dumps({"operation": "circle_area", "radius": 3}))) == {
        "success": True,
        "operation": "circle_area",
        "result": math.pi * 9,
    }
    assert json.loads(
        geometry_calculator(json.dumps({"operation": "evaluate", "expression": "(2 + 3) * 4"}))
    ) == {
        "success": True,
        "operation": "evaluate",
        "result": 20.0,
    }

    graph = create_graph(FakeToolCallingModel())
    initial_state: AgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "should_exit": False,
    }

    with patch(
        "builtins.input",
        side_effect=[
            "What is the area of a circle with radius 3?",
            "Evaluate (2 + 3) * 4.",
            "quit",
        ],
    ):
        output = graph.invoke(initial_state)

    tool_payloads = [
        json.loads(message.content)
        for message in output["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert any(
        payload.get("operation") == "circle_area" and payload.get("result") == math.pi * 9
        for payload in tool_payloads
    )
    assert any(
        payload.get("operation") == "evaluate" and payload.get("result") == 20.0
        for payload in tool_payloads
    )
    print("self_test_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run without calling the OpenAI API")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name to use for interactive mode",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Run with --self-test or configure the API key.")

    print("=" * 60)
    print("LangGraph Geometry Calculator Agent")
    print("=" * 60)

    graph = create_graph(OpenAIModel(args.model))
    save_graph_image(graph)

    initial_state: AgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "should_exit": False,
    }
    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
