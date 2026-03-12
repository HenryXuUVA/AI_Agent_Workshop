## Exercise1
LangGraph Multi-Turn Image Chat
================================================================================
Model: llava
Workflow:
  1. Upload one image by path
  2. Ask as many follow-up questions as you want
  3. Type 'exit' to finish
Commands:
  - verbose
  - quiet
  - exit

Images larger than 1024px on the longest side are resized when Pillow is installed.
If responses feel slow, reducing image resolution usually helps.
================================================================================

================================================================================
NODE: upload_image
================================================================================
Image path: D:\AgenticAI\Topic6\download.jfif
[SYSTEM] Loaded image: D:\AgenticAI\Topic6\download.jfif
[SYSTEM] Pillow is not installed, so the original image will be used.

================================================================================
NODE: summarize_image
================================================================================
[DEBUG] Image summary: The image depicts a person standing in front of a doorway on a residential street at night. The person is wearing a coat and a cap and is looking out, perhaps observing something or someone outside. The setting includes a well-lit porch with a railing, a tree, and a 
lamp. The street is quiet, with no other individuals or vehicles visible. The color palette is muted, with a focus on the person's attire and the artificial lighting from the porch and street lamps.

## Exercise2
Scanning video with LLaVA
Video: D:\AgenticAI\Topic6\test.mp4
Model: llava
Frame step: 1
Max image side: 512
Processed 1 frames (video frame 0/613)
Processed 25 frames (video frame 24/613)
Processed 50 frames (video frame 49/613)
Processed 75 frames (video frame 74/613)
Processed 100 frames (video frame 99/613)
Processed 125 frames (video frame 124/613)
Processed 150 frames (video frame 149/613)
Processed 175 frames (video frame 174/613)
Processed 200 frames (video frame 199/613)
Processed 225 frames (video frame 224/613)
Processed 250 frames (video frame 249/613)
Processed 275 frames (video frame 274/613)
Processed 300 frames (video frame 299/613)
Processed 325 frames (video frame 324/613)
Processed 350 frames (video frame 349/613)
Processed 375 frames (video frame 374/613)
Processed 400 frames (video frame 399/613)
Processed 425 frames (video frame 424/613)
Processed 450 frames (video frame 449/613)
Processed 475 frames (video frame 474/613)
Processed 500 frames (video frame 499/613)
Processed 525 frames (video frame 524/613)
Processed 550 frames (video frame 549/613)
Processed 575 frames (video frame 574/613)
Processed 600 frames (video frame 599/613)

Person entry/exit report
------------------------------------------------------------
Video frames: 614
Video duration: 00:00:20.500
1. enter: 00:00:00.100 (frame 3) | exit: 00:00:00.134 (frame 4)
2. enter: 00:00:00.234 (frame 7) | exit: 00:00:00.268 (frame 8)
3. enter: 00:00:00.435 (frame 13) | exit: 00:00:00.468 (frame 14)
4. enter: 00:00:00.803 (frame 24) | exit: 00:00:00.836 (frame 25)
5. enter: 00:00:00.936 (frame 28) | exit: 00:00:01.003 (frame 30)
6. enter: 00:00:01.037 (frame 31) | exit: 00:00:01.104 (frame 33)
7. enter: 00:00:01.605 (frame 48) | exit: 00:00:01.639 (frame 49)
8. enter: 00:00:01.672 (frame 50) | exit: 00:00:01.706 (frame 51)
9. enter: 00:00:01.739 (frame 52) | exit: 00:00:01.839 (frame 55)
10. enter: 00:00:01.873 (frame 56) | exit: 00:00:01.906 (frame 57)
11. enter: 00:00:01.973 (frame 59) | exit: 00:00:02.040 (frame 61)
12. enter: 00:00:02.073 (frame 62) | exit: 00:00:02.107 (frame 63)
13. enter: 00:00:02.140 (frame 64) | exit: 00:00:02.174 (frame 65)
14. enter: 00:00:02.207 (frame 66) | exit: 00:00:02.241 (frame 67)
15. enter: 00:00:02.274 (frame 68) | exit: 00:00:02.341 (frame 70)
16. enter: 00:00:02.408 (frame 72) | exit: 00:00:02.475 (frame 74)
17. enter: 00:00:02.508 (frame 75) | exit: 00:00:02.575 (frame 77)
18. enter: 00:00:02.642 (frame 79) | exit: 00:00:02.709 (frame 81)
19. enter: 00:00:02.742 (frame 82) | exit: 00:00:02.843 (frame 85)
20. enter: 00:00:02.876 (frame 86) | exit: 00:00:03.210 (frame 96)
21. enter: 00:00:03.378 (frame 101) | exit: 00:00:03.411 (frame 102)
22. enter: 00:00:03.445 (frame 103) | exit: 00:00:03.545 (frame 106)
23. enter: 00:00:03.645 (frame 109) | exit: 00:00:03.679 (frame 110)
24. enter: 00:00:03.745 (frame 112) | exit: 00:00:03.779 (frame 113)
25. enter: 00:00:03.812 (frame 114) | exit: 00:00:03.846 (frame 115)
26. enter: 00:00:03.946 (frame 118) | exit: 00:00:04.113 (frame 123)
27. enter: 00:00:04.147 (frame 124) | exit: 00:00:04.214 (frame 126)
28. enter: 00:00:04.782 (frame 143) | exit: 00:00:04.816 (frame 144)
29. enter: 00:00:04.849 (frame 145) | exit: 00:00:04.883 (frame 146)
30. enter: 00:00:04.949 (frame 148) | exit: 00:00:04.983 (frame 149)
31. enter: 00:00:05.016 (frame 150) | exit: 00:00:05.050 (frame 151)
32. enter: 00:00:05.585 (frame 167) | exit: 00:00:05.618 (frame 168)
33. enter: 00:00:05.785 (frame 173) | exit: 00:00:05.819 (frame 174)
34. enter: 00:00:05.886 (frame 176) | exit: 00:00:05.919 (frame 177)
35. enter: 00:00:06.086 (frame 182) | exit: 00:00:06.120 (frame 183)
36. enter: 00:00:06.956 (frame 208) | exit: 00:00:06.989 (frame 209)
37. enter: 00:00:07.090 (frame 212) | exit: 00:00:07.123 (frame 213)
38. enter: 00:00:07.157 (frame 214) | exit: 00:00:07.190 (frame 215)
39. enter: 00:00:07.993 (frame 239) | exit: 00:00:08.026 (frame 240)
40. enter: 00:00:08.060 (frame 241) | exit: 00:00:08.093 (frame 242)
41. enter: 00:00:08.394 (frame 251) | exit: 00:00:08.427 (frame 252)
42. enter: 00:00:08.494 (frame 254) | exit: 00:00:08.528 (frame 255)
43. enter: 00:00:08.929 (frame 267) | exit: 00:00:08.996 (frame 269)
44. enter: 00:00:09.063 (frame 271) | exit: 00:00:09.096 (frame 272)
45. enter: 00:00:09.130 (frame 273) | exit: 00:00:09.163 (frame 274)
46. enter: 00:00:09.197 (frame 275) | exit: 00:00:09.263 (frame 277)
47. enter: 00:00:09.330 (frame 279) | exit: 00:00:09.397 (frame 281)
48. enter: 00:00:09.431 (frame 282) | exit: 00:00:09.598 (frame 287)
49. enter: 00:00:09.665 (frame 289) | exit: 00:00:09.765 (frame 292)
50. enter: 00:00:09.832 (frame 294) | exit: 00:00:09.865 (frame 295)
51. enter: 00:00:09.899 (frame 296) | exit: 00:00:09.966 (frame 298)
52. enter: 00:00:09.999 (frame 299) | exit: 00:00:10.133 (frame 303)
53. enter: 00:00:10.200 (frame 305) | exit: 00:00:10.233 (frame 306)
54. enter: 00:00:10.300 (frame 308) | exit: 00:00:10.601 (frame 317)
55. enter: 00:00:10.635 (frame 318) | exit: 00:00:10.668 (frame 319)
56. enter: 00:00:10.735 (frame 321) | exit: 00:00:10.802 (frame 323)
57. enter: 00:00:10.902 (frame 326) | exit: 00:00:11.203 (frame 335)
58. enter: 00:00:11.270 (frame 337) | exit: 00:00:11.303 (frame 338)
59. enter: 00:00:11.337 (frame 339) | exit: 00:00:11.537 (frame 345)
60. enter: 00:00:11.671 (frame 349) | exit: 00:00:11.738 (frame 351)
61. enter: 00:00:12.006 (frame 359) | exit: 00:00:12.039 (frame 360)
62. enter: 00:00:12.273 (frame 367) | exit: 00:00:12.307 (frame 368)
63. enter: 00:00:12.340 (frame 369) | exit: 00:00:12.374 (frame 370)
64. enter: 00:00:12.875 (frame 385) | exit: 00:00:12.909 (frame 386)
65. enter: 00:00:13.143 (frame 393) | exit: 00:00:13.210 (frame 395)
66. enter: 00:00:13.644 (frame 408) | exit: 00:00:13.678 (frame 409)
67. enter: 00:00:14.413 (frame 431) | exit: 00:00:14.447 (frame 432)
68. enter: 00:00:14.614 (frame 437) | exit: 00:00:14.648 (frame 438)
69. enter: 00:00:14.882 (frame 445) | exit: 00:00:14.982 (frame 448)
70. enter: 00:00:15.049 (frame 450) | exit: 00:00:15.082 (frame 451)
71. enter: 00:00:15.116 (frame 452) | exit: 00:00:15.183 (frame 454)
72. enter: 00:00:15.216 (frame 455) | exit: 00:00:15.250 (frame 456)
73. enter: 00:00:15.484 (frame 463) | exit: 00:00:15.517 (frame 464)
74. enter: 00:00:15.952 (frame 477) | exit: 00:00:15.985 (frame 478)
75. enter: 00:00:16.286 (frame 487) | exit: 00:00:16.353 (frame 489)
76. enter: 00:00:16.520 (frame 494) | exit: 00:00:16.587 (frame 496)
77. enter: 00:00:16.621 (frame 497) | exit: 00:00:16.654 (frame 498)
78. enter: 00:00:16.988 (frame 508) | exit: 00:00:17.022 (frame 509)
79. enter: 00:00:17.256 (frame 516) | exit: 00:00:17.323 (frame 518)
80. enter: 00:00:17.724 (frame 530) | exit: 00:00:17.758 (frame 531)
81. enter: 00:00:17.992 (frame 538) | exit: 00:00:18.025 (frame 539)
82. enter: 00:00:18.226 (frame 545) | exit: 00:00:18.259 (frame 546)
83. enter: 00:00:18.426 (frame 551) | exit: 00:00:18.460 (frame 552)
84. enter: 00:00:18.661 (frame 558) | exit: 00:00:18.694 (frame 559)
85. enter: 00:00:18.761 (frame 561) | exit: 00:00:18.794 (frame 562)
86. enter: 00:00:18.828 (frame 563) | exit: 00:00:18.861 (frame 564)
87. enter: 00:00:18.895 (frame 565) | exit: 00:00:18.928 (frame 566)
88. enter: 00:00:18.962 (frame 567) | exit: 00:00:18.995 (frame 568)
89. enter: 00:00:19.028 (frame 569) | exit: 00:00:19.062 (frame 570)
90. enter: 00:00:19.095 (frame 571) | exit: 00:00:19.162 (frame 573)
91. enter: 00:00:19.196 (frame 574) | exit: 00:00:19.263 (frame 576)
92. enter: 00:00:19.329 (frame 578) | exit: 00:00:19.363 (frame 579)
93. enter: 00:00:19.396 (frame 580) | exit: 00:00:19.497 (frame 583)
94. enter: 00:00:19.530 (frame 584) | exit: 00:00:19.731 (frame 590)
95. enter: 00:00:19.764 (frame 591) | exit: 00:00:19.898 (frame 595)
96. enter: 00:00:19.931 (frame 596) | exit: 00:00:20.232 (frame 605)
97. enter: 00:00:20.266 (frame 606) | exit: 00:00:20.333 (frame 608)
98. enter: 00:00:20.366 (frame 609) | exit: 00:00:20.466 (frame 612)