"""
LangGraph multi-turn image chat agent backed by local Ollama LLaVA.

The program:
- Prompts once for an image path
- Optionally resizes large images before sending them to Ollama
- Builds a compact persistent summary of the image
- Keeps trimmed text history in LangGraph state
- Reattaches the image on each user turn so follow-up questions stay grounded
"""

import hashlib
import os
import uuid
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import ollama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import RemoveMessage, add_messages

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageOps = None


MODEL_NAME = "llava"
MAX_IMAGE_DIMENSION = 1024
MAX_HISTORY_MESSAGES = 24
MAX_CONTEXT_MESSAGES = 12
CACHE_DIR = Path(__file__).resolve().parent / ".image_cache"

IMAGE_SUMMARY_PROMPT = """Summarize this image for an assistant that will answer follow-up questions later.
Include the main subjects, key objects, spatial relationships, visible text, setting, colors,
and any uncertainty. Keep it under 180 words."""

ASSISTANT_PROMPT = """You are a helpful vision assistant.
Answer the user's question about the uploaded image.
Use the persistent image summary for continuity, but treat the attached image as the source of truth.
If you are uncertain, say what is unclear instead of inventing details."""


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    verbose: bool
    command: str | None
    pending_user_text: str
    image_path: str
    image_for_model: str
    image_summary: str
    original_size: tuple[int, int]
    working_size: tuple[int, int]
    resized: bool
    resize_available: bool


def new_message_id() -> str:
    return str(uuid.uuid4())


def normalize_text_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part.strip() for part in parts if part and str(part).strip()).strip()
    return str(content).strip()


def maybe_resize_image(image_path: str, max_dimension: int) -> dict[str, object]:
    source_path = Path(image_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    if Image is None or ImageOps is None:
        return {
            "image_path": str(source_path),
            "image_for_model": str(source_path),
            "original_size": (0, 0),
            "working_size": (0, 0),
            "resized": False,
            "resize_available": False,
        }

    with Image.open(source_path) as img:
        img = ImageOps.exif_transpose(img)
        original_size = img.size
        working = img.copy()

        resized = max(working.size) > max_dimension
        if not resized:
            return {
                "image_path": str(source_path),
                "image_for_model": str(source_path),
                "original_size": original_size,
                "working_size": working.size,
                "resized": False,
                "resize_available": True,
            }

        working.thumbnail((max_dimension, max_dimension))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha256(f"{source_path}:{max_dimension}".encode("utf-8")).hexdigest()[:16]
        if "A" in working.getbands():
            cached_path = CACHE_DIR / f"{digest}.png"
            working.save(cached_path, format="PNG")
        else:
            cached_path = CACHE_DIR / f"{digest}.jpg"
            working.convert("RGB").save(cached_path, format="JPEG", quality=85, optimize=True)

        return {
            "image_path": str(source_path),
            "image_for_model": str(cached_path),
            "original_size": original_size,
            "working_size": working.size,
            "resized": True,
            "resize_available": True,
        }


def prompt_for_image_path() -> str:
    while True:
        raw = input("Image path: ").strip().strip('"')
        if not raw:
            print("[SYSTEM] Please enter a path to an image file.")
            continue
        if os.path.exists(raw):
            return raw
        print(f"[SYSTEM] File not found: {raw}")


def build_ollama_messages(
    history: list[BaseMessage],
    user_text: str,
    image_summary: str,
    image_for_model: str,
) -> list[dict[str, object]]:
    recent_history = history[-MAX_CONTEXT_MESSAGES:]
    messages: list[dict[str, object]] = [
        {"role": "system", "content": ASSISTANT_PROMPT},
        {
            "role": "system",
            "content": f"Persistent image summary for conversation continuity:\n{image_summary}",
        },
    ]

    for message in recent_history:
        if isinstance(message, HumanMessage):
            messages.append({"role": "user", "content": normalize_text_content(message.content)})
        elif isinstance(message, AIMessage):
            messages.append({"role": "assistant", "content": normalize_text_content(message.content)})

    messages.append(
        {
            "role": "user",
            "content": user_text,
            "images": [image_for_model],
        }
    )
    return messages


def upload_image_node(state: ConversationState) -> ConversationState:
    if state.get("verbose", True):
        print("\n" + "=" * 80)
        print("NODE: upload_image")
        print("=" * 80)

    image_path = prompt_for_image_path()
    image_payload = maybe_resize_image(image_path, MAX_IMAGE_DIMENSION)

    print(f"[SYSTEM] Loaded image: {image_payload['image_path']}")
    if image_payload["resize_available"]:
        original_width, original_height = image_payload["original_size"]
        working_width, working_height = image_payload["working_size"]
        print(f"[SYSTEM] Original size: {original_width}x{original_height}")
        if image_payload["resized"]:
            print(
                f"[SYSTEM] Resized to {working_width}x{working_height} before upload to reduce latency."
            )
        else:
            print("[SYSTEM] No resizing needed.")
    else:
        print("[SYSTEM] Pillow is not installed, so the original image will be used.")

    return image_payload


def summarize_image_node(state: ConversationState) -> ConversationState:
    if state.get("verbose", True):
        print("\n" + "=" * 80)
        print("NODE: summarize_image")
        print("=" * 80)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": IMAGE_SUMMARY_PROMPT},
            {
                "role": "user",
                "content": "Summarize the uploaded image for future conversational context.",
                "images": [state["image_for_model"]],
            },
        ],
    )
    image_summary = normalize_text_content(response["message"]["content"])

    if state.get("verbose", True):
        print(f"[DEBUG] Image summary: {image_summary}")

    return {"image_summary": image_summary}


def input_node(state: ConversationState) -> ConversationState:
    if state.get("verbose", True):
        print("\n" + "=" * 80)
        print("NODE: input")
        print("=" * 80)

    user_input = input("\nYou: ").strip()

    if not user_input:
        print("[SYSTEM] Please enter a question about the image.")
        return {"command": "retry"}

    lowered = user_input.lower()
    if lowered in {"quit", "exit"}:
        return {"command": "exit"}
    if lowered == "verbose":
        print("[SYSTEM] Verbose mode enabled")
        return {"command": "retry", "verbose": True}
    if lowered == "quiet":
        print("[SYSTEM] Verbose mode disabled")
        return {"command": "retry", "verbose": False}

    return {"command": None, "pending_user_text": user_input}


def route_after_input(state: ConversationState) -> Literal["call_model", "input", "end"]:
    if state.get("command") == "exit":
        if state.get("verbose", True):
            print("[DEBUG] Routing to END")
        return "end"
    if state.get("verbose", True):
        if state.get("command") is None:
            print("[DEBUG] Routing to call_model")
        else:
            print("[DEBUG] Routing to input")
    return "call_model" if state.get("command") is None else "input"


def call_model_node(state: ConversationState) -> ConversationState:
    if state.get("verbose", True):
        print("\n" + "=" * 80)
        print("NODE: call_model")
        print("=" * 80)
        print(f"[DEBUG] History messages in state: {len(state['messages'])}")

    ollama_messages = build_ollama_messages(
        history=state["messages"],
        user_text=state["pending_user_text"],
        image_summary=state["image_summary"],
        image_for_model=state["image_for_model"],
    )
    response = ollama.chat(model=MODEL_NAME, messages=ollama_messages)
    answer = normalize_text_content(response["message"]["content"])

    if state.get("verbose", True):
        print(f"[DEBUG] Response preview: {answer[:120]}")

    return {
        "messages": [
            HumanMessage(content=state["pending_user_text"], id=new_message_id()),
            AIMessage(content=answer, id=new_message_id()),
        ],
        "pending_user_text": "",
        "command": None,
    }


def output_node(state: ConversationState) -> ConversationState:
    if state.get("verbose", True):
        print("\n" + "=" * 80)
        print("NODE: output")
        print("=" * 80)

    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            print(f"\nAssistant: {normalize_text_content(message.content)}")
            break

    return {}


def trim_history_node(state: ConversationState) -> ConversationState:
    messages = state["messages"]
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return {}

    removals = [RemoveMessage(id=message.id) for message in messages[:-MAX_HISTORY_MESSAGES] if message.id]

    if state.get("verbose", True):
        print(f"[DEBUG] Trimming {len(removals)} older messages from state history")

    return {"messages": removals}


def create_graph():
    workflow = StateGraph(ConversationState)

    workflow.add_node("upload_image", upload_image_node)
    workflow.add_node("summarize_image", summarize_image_node)
    workflow.add_node("input", input_node)
    workflow.add_node("call_model", call_model_node)
    workflow.add_node("output", output_node)
    workflow.add_node("trim_history", trim_history_node)

    workflow.set_entry_point("upload_image")
    workflow.add_edge("upload_image", "summarize_image")
    workflow.add_edge("summarize_image", "input")
    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {
            "call_model": "call_model",
            "input": "input",
            "end": END,
        },
    )
    workflow.add_edge("call_model", "output")
    workflow.add_edge("output", "trim_history")
    workflow.add_edge("trim_history", "input")

    return workflow.compile()


def main() -> None:
    print("=" * 80)
    print("LangGraph Multi-Turn Image Chat")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print("Workflow:")
    print("  1. Upload one image by path")
    print("  2. Ask as many follow-up questions as you want")
    print("  3. Type 'exit' to finish")
    print("Commands:")
    print("  - verbose")
    print("  - quiet")
    print("  - exit")
    print(f"\nImages larger than {MAX_IMAGE_DIMENSION}px on the longest side are resized when Pillow is installed.")
    print("If responses feel slow, reducing image resolution usually helps.")
    print("=" * 80)

    app = create_graph()
    initial_state: ConversationState = {
        "messages": [],
        "verbose": True,
        "command": None,
        "pending_user_text": "",
        "image_path": "",
        "image_for_model": "",
        "image_summary": "",
        "original_size": (0, 0),
        "working_size": (0, 0),
        "resized": False,
        "resize_available": False,
    }

    try:
        app.invoke(initial_state)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user.")

    print("[SYSTEM] Conversation ended.")


if __name__ == "__main__":
    main()
