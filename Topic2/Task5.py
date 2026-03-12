# Task5.py
# LangGraph agent using Message API (system/user/assistant/tool-style messages).
# Qwen routing is intentionally disabled; all non-command input goes to Llama.

import argparse
from typing import Annotated, List
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


class AgentState(dict):
    """Typed-like state container for LangGraph Message API usage."""


# Typed keys for readability; LangGraph uses TypedDict-like annotations.
class AgentStateType(dict):
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


def _trace(state: dict, message: str) -> None:
    if state.get("trace_enabled", False):
        print(f"[TRACE] {message}")


def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        role = "user"
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "tool"
        lines.append(f"{role}: {msg.content}")
    lines.append("assistant:")
    return "\n".join(lines)


def create_graph(llama_llm):
    def get_user_input(state: dict) -> dict:
        _trace(
            state,
            "node=get_user_input entered "
            f"(should_exit={state.get('should_exit')}, skip_llm={state.get('skip_llm')})",
        )

        print("\n" + "=" * 50)
        print(
            "Enter your text ('Hey Qwen ...' is disabled and will use Llama, "
            "'verbose'/'quiet' to toggle tracing, or 'quit' to exit):"
        )
        print("=" * 50)
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

        if normalized.startswith("hey qwen"):
            print("Qwen is disabled in Task5; using Llama instead.")

        return {
            "skip_llm": False,
            "messages": [HumanMessage(content=user_input)],
        }

    def call_llama(state: dict) -> dict:
        _trace(state, "node=call_llama entered")
        print("\nRunning Llama...")

        prompt = _messages_to_prompt(state.get("messages", []))
        response = llama_llm.invoke(prompt)

        if not isinstance(response, str):
            response = str(response)

        _trace(state, f"node=call_llama completed (response_len={len(response)})")
        return {
            "active_model": "llama",
            "messages": [AIMessage(content=response)],
        }

    def print_response(state: dict) -> dict:
        _trace(
            state,
            "node=print_response entered "
            f"(active_model={state.get('active_model', '')})",
        )

        latest_ai = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                latest_ai = msg.content
                break

        print("\n" + "-" * 50)
        print("Llama Response:")
        print("-" * 50)
        print(latest_ai)
        return {}

    def route_after_input(state: dict) -> str:
        _trace(
            state,
            "route_after_input "
            f"(should_exit={state.get('should_exit')}, skip_llm={state.get('skip_llm')})",
        )

        if state.get("should_exit", False):
            return END

        if state.get("skip_llm", False):
            return "get_user_input"

        # Qwen is disabled for this task: always use Llama.
        return "call_llama"

    graph_builder = StateGraph(AgentStateType)
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_llama": "call_llama",
            END: END,
        },
    )
    graph_builder.add_edge("call_llama", "print_response")
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


class _FakeLLM:
    def invoke(self, prompt: str) -> str:
        last_user = ""
        for line in prompt.splitlines()[::-1]:
            if line.startswith("user:"):
                last_user = line.split("user:", 1)[1].strip()
                break
        return f"fake-llama-response to: {last_user}"


def run_self_test() -> None:
    graph = create_graph(_FakeLLM())

    base_state = {
        "messages": [SystemMessage(content="You are a helpful assistant.")],
        "should_exit": False,
        "skip_llm": False,
        "trace_enabled": False,
        "active_model": "",
    }

    # Test 1: normal input should go to Llama, then quit.
    with patch("builtins.input", side_effect=["hello", "quit"]):
        out = graph.invoke(base_state)
    assert out.get("active_model") == "llama", "Expected llama route for normal input"
    assert any(isinstance(m, AIMessage) for m in out.get("messages", [])), "Expected AIMessage in state"

    # Test 2: "Hey Qwen..." should still route to Llama (Qwen disabled), then quit.
    with patch("builtins.input", side_effect=["Hey Qwen, what time is it?", "quit"]):
        out2 = graph.invoke(base_state)
    assert out2.get("active_model") == "llama", "Qwen must be disabled in Task5"

    print("self_test_ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run local graph tests without loading HF models")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    print("=" * 50)
    print("LangGraph Message API Agent (Qwen disabled)")
    print("=" * 50)
    print()

    device = get_device()
    llama_llm = create_model_llm("meta-llama/Llama-3.2-1B-Instruct", device)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state = {
        "messages": [SystemMessage(content="You are a helpful assistant.")],
        "should_exit": False,
        "skip_llm": False,
        "trace_enabled": False,
        "active_model": "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
