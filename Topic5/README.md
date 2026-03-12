# Topic 5: Retrieval-Augmented Generation (RAG)

This directory contains modularized Python scripts for Topic 5. Each exercise extends the same RAG workflow and evaluates a different retrieval or prompting choice.

## Project Structure

```text
.
├── manual_rag_pipeline_universal.ipynb
├── Topic5.ipynb
├── Exercise_2.py
├── Exercise_6.py
├── Exercise_7.py
├── Exercise_8.py
├── Exercise_9.py
├── Exercise_10.py
├── Exercise_11.py
├── Output.txt
└── README.md
```

## Exercise 1: Open Model RAG vs. No RAG Comparison

- Without RAG, the model hallucinated unsafe and incorrect values.
- With RAG, answers were grounded in the retrieved manual text.
- Some general-knowledge questions were still answered reasonably without RAG.

## Exercise 2: Open Model + RAG vs. Large Model Comparison

- GPT-4o Mini avoided hallucinations more reliably than Qwen 2.5 1.5B.
- It handled general-knowledge questions better, even when strict document grounding was missing.

## Exercise 3: Open Model + RAG vs. Frontier Chat Model

- Strong general models worked well on broad public knowledge.
- The RAG system was better when answers needed to match a specific source document exactly.
- This shows that RAG is most valuable for fixed, internal, or version-specific documents.

## Exercise 4: Effect of Top-K Retrieval Count

- Retrieval quality stopped improving once `k` went beyond 5.
- Large `k` values added confusing and irrelevant context.
- Chunk size and `k` need to be tuned together.

## Exercise 5: Handling Unanswerable Questions

- The stricter prompt made the model much more willing to say it did not know.
- Retrieved context sometimes encouraged hallucination when the prompt forced an answer.

## Exercise 6: Query Phrasing Sensitivity

- Formal and keyword-focused queries retrieved the best chunks.
- Informal questions were less reliable for vector retrieval.
- A query rewriting step would likely improve robustness.

## Exercise 7: Chunk Overlap Experiment

- Moderate overlap improved retrieval across chunk boundaries.
- Larger overlap increased redundancy and cost.
- Gains diminished quickly after the first useful overlap increase.

## Exercise 8: Chunk Size Experiment

- Small chunks improved precision but hurt completeness.
- Large chunks improved completeness but introduced more noise.
- The likely sweet spot for this corpus was between 512 and 1024 characters.

## Exercise 9: Retrieval Score Analysis

- High top scores usually correlated with strong grounding.
- Low top scores correlated with hallucination risk.
- A filtering threshold around 0.3 to 0.4 looked reasonable for this dataset.

## Exercise 10: Prompt Template Variations

- Strict grounding produced the most accurate answers.
- Structured output produced the most useful answers for answerable questions.
- There is a clear trade-off between helpfulness and strict grounding.

## Exercise 11: Cross-Document Synthesis

- The model could combine information across chunks, but success depended on retrieval quality.
- Too little context caused omissions, while too much context caused confusion.
- Contradictory chunks introduced noise and inconsistency.

## Group

- Chenxu Li - jnr2jp
- Wenhao Xu - wx8mcm

## Project Topics

- [Topic 1: Running an LLM](../../tree/Topic-1-Running-an-LLM)
- [Topic 2: Agent Orchestration Frameworks](../../tree/Topic-2-Agent-Orchestration-Frameworks)
- [Topic 3: Agent Tool Use](../../tree/Topic-3-Agent-Tool-Use)
- [Topic 4: Exploring Tools](../../tree/Topic-4-Exploring-Tools)
- [Topic 5: RAG](../../tree/Topic-5-RAG)
- [Topic 6: VLM](../../tree/Topic-6-VLM)
- [Topic 7: MCP and A2A](../../tree/Topic-7-MCP-and-A2A)
