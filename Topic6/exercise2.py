"""
Scan a video with LLaVA and report when a person enters or exits the scene.

Default behavior checks every frame. This is accurate but slow because every frame
is sent to the vision model. If runtime is too high, reduce `--max-side` or raise
`--frame-step`.
"""

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import ollama


MODEL_NAME = "llava"
PERSON_PROMPT = (
    "Look at this video frame and decide whether at least one person is visible anywhere "
    "in the scene. Respond with exactly YES or NO. Do not add any other words."
)


@dataclass
class SceneInterval:
    enter_frame: int
    enter_time: float
    exit_frame: int | None = None
    exit_time: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect when a person enters or exits a video using Ollama LLaVA."
    )
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Ollama model to use (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame. Default is 1, which checks every frame.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Resize the longest image side before inference. Smaller is faster.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality used for temporary frame files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame model decisions.",
    )
    return parser.parse_args()


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def resize_frame(frame, max_side: int):
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return frame

    scale = max_side / longest_side
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def write_temp_frame(frame, temp_path: Path, max_side: int, jpeg_quality: int) -> None:
    resized = resize_frame(frame, max_side)
    success = cv2.imwrite(
        str(temp_path),
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise RuntimeError(f"Failed to write temporary frame to {temp_path}")


def llava_says_person(image_path: Path, model: str) -> tuple[bool, str]:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": PERSON_PROMPT, "images": [str(image_path)]},
        ],
    )
    answer = response["message"]["content"].strip()
    normalized = answer.upper()

    if normalized.startswith("YES"):
        return True, answer
    if normalized.startswith("NO"):
        return False, answer

    compact = normalized.replace(".", "").replace("!", "").strip()
    if compact == "YES":
        return True, answer
    if compact == "NO":
        return False, answer

    raise ValueError(f"Unexpected LLaVA response: {answer!r}")


def frame_time(frame_index: int, fps: float) -> float:
    if fps <= 0:
        return float(frame_index)
    return frame_index / fps


def scan_video(
    video_path: Path,
    model: str,
    frame_step: int,
    max_side: int,
    jpeg_quality: int,
    verbose: bool,
) -> tuple[list[SceneInterval], float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_step < 1:
        raise ValueError("--frame-step must be at least 1")

    intervals: list[SceneInterval] = []
    person_in_scene = False
    current_interval: SceneInterval | None = None
    processed_frames = 0
    last_processed_frame = -1

    with tempfile.TemporaryDirectory(prefix="llava_frames_") as temp_dir:
        temp_frame_path = Path(temp_dir) / "frame.jpg"
        frame_index = 0

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index % frame_step != 0:
                    frame_index += 1
                    continue

                write_temp_frame(frame, temp_frame_path, max_side=max_side, jpeg_quality=jpeg_quality)
                has_person, raw_answer = llava_says_person(temp_frame_path, model=model)

                timestamp = frame_time(frame_index, fps)
                processed_frames += 1
                last_processed_frame = frame_index

                if verbose:
                    print(
                        f"frame={frame_index} time={format_timestamp(timestamp)} "
                        f"person={has_person} raw={raw_answer!r}"
                    )
                elif processed_frames == 1 or processed_frames % 25 == 0:
                    print(
                        f"Processed {processed_frames} frames "
                        f"(video frame {frame_index}/{max(total_frames - 1, 0)})"
                    )

                if has_person and not person_in_scene:
                    current_interval = SceneInterval(
                        enter_frame=frame_index,
                        enter_time=timestamp,
                    )
                    intervals.append(current_interval)
                    person_in_scene = True
                elif not has_person and person_in_scene:
                    if current_interval is not None:
                        current_interval.exit_frame = frame_index
                        current_interval.exit_time = timestamp
                    person_in_scene = False
                    current_interval = None

                frame_index += 1
        finally:
            capture.release()

    video_duration = frame_time(total_frames - 1, fps) if total_frames > 0 else 0.0
    if person_in_scene and current_interval is not None:
        end_frame = last_processed_frame if last_processed_frame >= 0 else 0
        current_interval.exit_frame = None
        current_interval.exit_time = None
        if verbose:
            print(
                "Person still present at end of video "
                f"(last processed frame {end_frame}, {format_timestamp(video_duration)})"
            )

    return intervals, video_duration, total_frames


def print_report(intervals: list[SceneInterval], video_duration: float, total_frames: int) -> None:
    print("\nPerson entry/exit report")
    print("-" * 60)
    print(f"Video frames: {total_frames}")
    print(f"Video duration: {format_timestamp(video_duration)}")

    if not intervals:
        print("No person detected in the processed frames.")
        return

    for index, interval in enumerate(intervals, start=1):
        enter_text = format_timestamp(interval.enter_time)
        if interval.exit_time is None:
            exit_text = "still present at end of video"
        else:
            exit_text = format_timestamp(interval.exit_time)

        print(
            f"{index}. enter: {enter_text} (frame {interval.enter_frame}) | "
            f"exit: {exit_text}"
            + (
                ""
                if interval.exit_frame is None
                else f" (frame {interval.exit_frame})"
            )
        )


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("Scanning video with LLaVA")
    print(f"Video: {video_path}")
    print(f"Model: {args.model}")
    print(f"Frame step: {args.frame_step}")
    print(f"Max image side: {args.max_side}")

    intervals, video_duration, total_frames = scan_video(
        video_path=video_path,
        model=args.model,
        frame_step=args.frame_step,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        verbose=args.verbose,
    )
    print_report(intervals, video_duration, total_frames)


if __name__ == "__main__":
    main()
