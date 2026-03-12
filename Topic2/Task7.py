# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a very simple agent.
# It writes to stdout and asks the user to enter a line of text through stdin.
# It passes the line to the LLM llama-3.2-1B-Instruct, then prints the
# what the LLM returns as text to stdout.
# The LLM should use Cuda if available, if not then if mps is available then use that,
# otherwise use cpu.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write a image of the graph to the file lg_graph.png
# The program gets the LLM llama-3.2-1B-Instruct from Hugging Face and wraps
# it for LangChain using HuggingFacePipeline.
# The code is commented in detail so a reader can understand each step.

# Import necessary libraries
import os
import pickle
import threading
import time
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# Prefer sqlite checkpointer when available; fallback keeps script runnable.
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    CHECKPOINTER_MODE = "sqlite"
except ImportError:
    from langgraph.checkpoint.memory import InMemorySaver
    CHECKPOINTER_MODE = "memory"


class FileCheckpointSaver(InMemorySaver):
    """
    Disk-backed checkpointer fallback when SqliteSaver is unavailable.
    Persists InMemorySaver internals to a local file on every checkpoint write.
    """
    _global_lock = threading.Lock()

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        with self._global_lock:
            if not os.path.exists(self.file_path):
                return
            with open(self.file_path, "rb") as f:
                payload = pickle.load(f)
        loaded_storage = payload.get("storage", {})
        self.storage = defaultdict(lambda: defaultdict(dict))
        for thread_id, ns_map in loaded_storage.items():
            self.storage[thread_id] = defaultdict(dict, ns_map)
        self.writes = defaultdict(dict, payload.get("writes", {}))
        self.blobs = payload.get("blobs", {})

    def _plain_storage(self) -> dict:
        plain = {}
        for thread_id, ns_map in self.storage.items():
            plain[thread_id] = dict(ns_map)
        return plain

    def _persist_to_disk(self) -> None:
        with self._global_lock:
            os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
            for attempt in range(5):
                try:
                    with open(self.file_path, "wb") as f:
                        pickle.dump(
                            {
                                "storage": self._plain_storage(),
                                "writes": dict(self.writes),
                                "blobs": dict(self.blobs),
                            },
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05)

    def put(self, config, checkpoint, metadata, new_versions):
        out_config = super().put(config, checkpoint, metadata, new_versions)
        self._persist_to_disk()
        return out_config

    def put_writes(self, config, writes, task_id, task_path=""):
        super().put_writes(config, writes, task_id, task_path)
        self._persist_to_disk()


# Determine the best available device for inference
# Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"


# =============================================================================
# STATE DEFINITION
# =============================================================================
# The state is a TypedDict that flows through all nodes in the graph.
# Each node can read from and write to specific fields in the state.
# LangGraph automatically merges the returned dict from each node into the state.


class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - messages: List of messages maintaining chat history (user, ai)
    - user_input: The text entered by the user (set by get_user_input node)
    - should_exit: Boolean flag indicating if user wants to quit (set by get_user_input node)
    - verbose: Boolean flag for trace logging
    """

    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    should_exit: bool
    verbose: bool
    # Track the last active model for routing purposes if needed.
    last_model: Optional[str]


def create_specific_llm(model_id):
    """
    Create and configure a specific LLM using HuggingFace's transformers library.
    """
    device = get_device()

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

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
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    print(f"Model {model_id} loaded successfully!")
    return llm


def format_history_for_model(messages: List[BaseMessage], current_model_name: str) -> List[dict]:
    """
    Transforms chat history into model-specific role format.

    - Human messages -> user
    - Current model messages -> assistant
    - Other model messages -> user
    """
    formatted_messages = []

    participants = "Human, Llama, and Qwen"
    system_prompt = (
        f"You are {current_model_name}. "
        f"You are participating in a conversation between {participants}."
    )
    formatted_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        content = msg.content
        if content.startswith("Human:"):
            role = "user"
        elif content.startswith(f"{current_model_name}:"):
            role = "assistant"
        else:
            role = "user"
        formatted_messages.append({"role": role, "content": content})

    return formatted_messages


def create_graph(llama_llm, qwen_llm, checkpointer):
    """
    Create the LangGraph state graph.
    """

    def get_user_input(state: AgentState) -> dict:
        if state.get("verbose", False):
            print(f"[Trace] Entering node: get_user_input. Current state keys: {list(state.keys())}")

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input().strip()

        verbose = state.get("verbose", False)
        if user_input == "verbose":
            verbose = True
            print("Verbose mode enabled.")
        elif user_input == "quiet":
            verbose = False
            print("Quiet mode enabled.")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
                "verbose": verbose,
            }

        messages_update = []
        if user_input and user_input not in ["verbose", "quiet"]:
            messages_update = [HumanMessage(content=f"Human: {user_input}")]

        return {
            "user_input": user_input,
            "should_exit": False,
            "messages": messages_update,
            "verbose": verbose,
        }

    def call_llama(state: AgentState) -> dict:
        if state.get("verbose", False):
            print(f"[Trace] Entering node: call_llama. History length: {len(state['messages'])}")

        formatted_messages = format_history_for_model(state["messages"], "Llama")

        prompt = llama_llm.pipeline.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print("\nLlama is processing...")
        response = llama_llm.invoke(prompt)

        return {
            "messages": [AIMessage(content=f"Llama: {response.strip()}")],
            "last_model": "llama",
        }

    def call_qwen(state: AgentState) -> dict:
        if state.get("verbose", False):
            print(f"[Trace] Entering node: call_qwen. History length: {len(state['messages'])}")

        formatted_messages = format_history_for_model(state["messages"], "Qwen")

        prompt = qwen_llm.pipeline.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print("\nQwen is processing...")
        response = qwen_llm.invoke(prompt)

        return {
            "messages": [AIMessage(content=f"Qwen: {response.strip()}")],
            "last_model": "qwen",
        }

    def print_response(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[Trace] Entering node: print_response.")

        if state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage):
                print("\n" + "-" * 50)
                print("Response:")
                print("-" * 50)
                print(last_message.content)

        return {}

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END

        if (not state["user_input"]) or state["user_input"] in ["verbose", "quiet"]:
            return "get_user_input"

        if state["user_input"].lower().startswith("hey qwen"):
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
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph


def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")


def resume_or_start(graph, config, initial_state: AgentState):
    """
    Resume existing thread if checkpoints exist, else start from initial state.
    """
    snapshot = graph.get_state(config)
    has_messages = bool(snapshot and snapshot.values and snapshot.values.get("messages"))
    if has_messages:
        values = snapshot.values or {}
        # If the last saved turn exited (e.g. user typed quit), clear exit flag
        # so the next launch returns to the input prompt with full history intact.
        if values.get("should_exit", False):
            print(
                f"Found ended session for thread_id={config['configurable']['thread_id']}; "
                "re-opening conversation from saved history..."
            )
            return graph.invoke(
                {"should_exit": False, "user_input": "", "verbose": values.get("verbose", False)},
                config=config,
            )

        print(f"Resuming saved session for thread_id={config['configurable']['thread_id']}...")
        return graph.invoke(None, config=config)
    print(f"Starting new session for thread_id={config['configurable']['thread_id']}...")
    return graph.invoke(initial_state, config=config)


def main():
    print("=" * 50)
    print("LangGraph Multi-Agent Chat (Llama 3.2 & Qwen 2.5 Shared History)")
    print("=" * 50)
    print()

    print("Initializing Llama...")
    llama_llm = create_specific_llm("meta-llama/Llama-3.2-1B-Instruct")

    print("\nInitializing Qwen...")
    qwen_llm = create_specific_llm("Qwen/Qwen2.5-1.5B-Instruct")

    print("\nCreating LangGraph...")

    if CHECKPOINTER_MODE == "sqlite":
        with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
            graph = create_graph(llama_llm, qwen_llm, checkpointer)
            print("Graph created successfully!")

            print("\nSaving graph visualization...")
            save_graph_image(graph)

            initial_state: AgentState = {
                "messages": [],
                "user_input": "",
                "should_exit": False,
                "verbose": False,
                "last_model": None,
            }

            config = {"configurable": {"thread_id": "1"}}
            resume_or_start(graph, config, initial_state)
    else:
        print("Warning: sqlite checkpointer not available; using file-backed fallback checkpointing.")
        checkpointer = FileCheckpointSaver("checkpoints_fallback.bin")
        graph = create_graph(llama_llm, qwen_llm, checkpointer)
        print("Graph created successfully!")

        print("\nSaving graph visualization...")
        save_graph_image(graph)

        initial_state: AgentState = {
            "messages": [],
            "user_input": "",
            "should_exit": False,
            "verbose": False,
            "last_model": None,
        }

        config = {"configurable": {"thread_id": "1"}}
        resume_or_start(graph, config, initial_state)


if __name__ == "__main__":
    main()
