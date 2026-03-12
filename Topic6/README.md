# Topic 6

## Exercise 1

```text
LangGraph Multi-Turn Image Chat
Model: llava

Workflow:
  1. Upload one image by path
  2. Ask follow-up questions
  3. Type 'exit' to finish

Commands:
  - verbose
  - quiet
  - exit

Example run:
Image path: D:\AgenticAI\Topic6\download.jfif
[SYSTEM] Loaded image: D:\AgenticAI\Topic6\download.jfif
[SYSTEM] Pillow is not installed, so the original image will be used.

[DEBUG] Image summary:
The image depicts a person standing in front of a doorway on a residential street at night.
```

## Exercise 2

```text
Scanning video with LLaVA
Video: D:\AgenticAI\Topic6\test.mp4
Model: llava
Frame step: 1
Max image side: 512

Person entry/exit report
Video frames: 614
Video duration: 00:00:20.500

Example intervals:
1. enter: 00:00:00.100 (frame 3) | exit: 00:00:00.134 (frame 4)
2. enter: 00:00:00.234 (frame 7) | exit: 00:00:00.268 (frame 8)
3. enter: 00:00:00.435 (frame 13) | exit: 00:00:00.468 (frame 14)
...
98. enter: 00:00:20.366 (frame 609) | exit: 00:00:20.466 (frame 612)
```

The raw frame-by-frame output showed heavy flicker, which suggests the detector should be smoothed before treating those boundaries as final scene-change events.
