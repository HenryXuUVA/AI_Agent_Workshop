"""
Educational Video Analyzer

Student provides a YouTube URL.
The agent fetches the transcript and generates:
- a summary
- key concepts
- quiz questions with answers
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict
from urllib.parse import parse_qs, urlparse

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:  # pragma: no cover - depends on local environment
    YouTubeTranscriptApi = None


class QuizQuestion(TypedDict):
    question: str
    answer: str


class VideoAnalysis(TypedDict):
    summary: str
    key_concepts: list[str]
    quiz_questions: list[QuizQuestion]


def extract_video_id(video_url: str) -> str:
    """Extract a YouTube video ID from a standard YouTube URL."""
    parsed = urlparse(video_url.strip())
    host = parsed.netloc.lower().replace("www.", "")
    path_parts = [part for part in parsed.path.split("/") if part]

    if host == "youtu.be" and path_parts:
        return path_parts[0]

    if host in {"youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
            if video_id:
                return video_id

        if path_parts and path_parts[0] in {"embed", "shorts", "live"} and len(path_parts) > 1:
            return path_parts[1]

    raise ValueError(
        "Could not extract a YouTube video ID from the provided URL. "
        "Use a standard youtube.com or youtu.be link."
    )


@tool
def get_youtube_transcript(video_url: str) -> str:
    """Fetch the transcript of a YouTube video from its URL."""
    if YouTubeTranscriptApi is None:
        raise RuntimeError(
            "youtube_transcript_api is not installed. Run "
            "'pip install youtube-transcript-api' first."
        )

    video_id = extract_video_id(video_url)
    transcript = YouTubeTranscriptApi().fetch(video_id)
    transcript_text = " ".join(
        snippet.text.strip() for snippet in transcript if snippet.text.strip()
    )

    if not transcript_text:
        raise RuntimeError("Transcript was found but contained no usable text.")

    return transcript_text


def build_agent():
    """Create the educational video analysis agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    analysis_prompt = (
        "You are an educational video analyzer. "
        "Always use the get_youtube_transcript tool before answering. "
        "After reading the transcript, produce a concise study guide. "
        "Focus on accurate explanation, not hype."
    )

    return create_agent(
        llm,
        [get_youtube_transcript],
        system_prompt=analysis_prompt,
        response_format=VideoAnalysis,
    )


def analyze_video(video_url: str) -> VideoAnalysis:
    """Run the agent and return a structured analysis."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    agent = build_agent()
    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    (
                        "Analyze this educational video URL. "
                        "Use the transcript tool, then generate a summary, "
                        f"key concepts, and quiz questions: {video_url}"
                    ),
                )
            ]
        }
    )
    return result["structured_response"]


def print_analysis(analysis: VideoAnalysis) -> None:
    """Pretty-print the structured analysis."""
    print("\nSummary")
    print("-" * 60)
    print(analysis["summary"])

    print("\nKey Concepts")
    print("-" * 60)
    for index, concept in enumerate(analysis["key_concepts"], start=1):
        print(f"{index}. {concept}")

    print("\nQuiz Questions")
    print("-" * 60)
    for index, item in enumerate(analysis["quiz_questions"], start=1):
        print(f"{index}. {item['question']}")
        print(f"   Answer: {item['answer']}")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        video_url = input("Enter a YouTube video URL: ").strip()

    if not video_url:
        raise SystemExit("No YouTube URL provided.")

    try:
        analysis = analyze_video(video_url)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print_analysis(analysis)


if __name__ == "__main__":
    main()
