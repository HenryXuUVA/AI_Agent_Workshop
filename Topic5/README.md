# Topic 5: Retrieval-Augmented Generation (RAG)

This repository contains modularized Python scripts for **Topic 5: RAG**. Functionality added separately based on the exercises, building upon the existing code.


---

## Project Structure

```text
.
├── manual_rag_pipeline_universal.ipynb       # Original notebook
├── Topic5.ipynb                              # Full notebook
├── Exercise_2.py                             # Open Model + RAG vs. Large Model Comparison
├── Exercise_6.py                             # Query Phrasing Sensitivity
├── Exercise_7.py                             # Chunk Overlap Experiment
├── Exercise_8.py                             # Chunk Size Experiment
├── Exercise_9.py                             # Retrieval Score Analysis
├── Exercise_10.py                            # Prompt Template Variations
├── Exercise_11.py                            # Cross-Document/Cross-Chunk Synthesis
├── Output.txt                                # Output
└── README.md                                 # Documentation

```




## Exercise 1: Open Model RAG vs. No RAG Comparison
### *Does the model hallucinate specific values without RAG?*
Yes, the model creates serious illusions when there is a lack of RAG context. For example, it suggests using a mixture of "80% gasoline and 20% distilled water" as engine oil.

### *Does RAG ground the answers in the actual manual?*
Yes, after introducing RAG, the model's answers are clearly based on actual text fragments. It accurately extracts specific details from the manual.

### *Are there questions where the model's general knowledge is actually correct?*
Yes, some common sense about the model is roughly correct when there is no RAG.

## Exercise 2: Open Model + RAG vs. Large Model Comparison
### *Does GPT 4o Mini do a better job than Qwen 2.5 1.5B in avoiding hallucinations?*
I believe so. The GPT-4o Mini performs significantly better in avoiding hallucinations. For unknown information, the model explicitly acknowledges knowledge gaps, completely avoiding hallucinations related to specific events.

### *Which questions does GPT 4o Mini answer correctly?  Compare the cut-off date of GPT 4o Mini pre-training and the age of the Model T Ford and Congressional Record corpora.*
From a documentation perspective, the GPT-4o Mini doesn't answer the question correctly in a strict sense, because it lacks documentation. However, it answers the question relatively correctly at the level of "general knowledge."

## Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model
In this exercise, I used the Gemini 3.1 Pro for comparison.
### *Where does the frontier model's general knowledge succeed?*
I believe we've achieved great success with the Model T Ford issue. The large model incorporates a wealth of online repair knowledge, providing safe and standard-compliant repair advice.

### *When did the frontier model appear to be using live web search to help answer your questions?*
I believe this applies when dealing with questions that have a clear, recent timestamp. For example, the model knows that its pre-training data may not cover specific congressional debates from a few weeks ago, so it must rely on a search engine to retrieve the bill names and specific information.

### *Where does your RAG system provide more accurate, specific answers?*
The RAG system directly quotes the fixed terminology from the manual, while the frontier big model provides a more modern and universal alternative, rather than the "original words" from the 1920 manual.

### *What does this tell you about when RAG adds value vs. when a powerful model suffices?*
I believe that large models can handle questions that fall into general domains, are common sense, or are timely facts that have been widely discussed on the public internet, without the need for RAG. However, RAG is irreplaceable when you need to answer questions strictly based on a specific version of internal documents (such as an operating manual from a specific year or unpublished meeting minutes), or when the questions are extremely subjective and limited to a specific text.

## Exercise 4: Effect of Top-K Retrieval Count

### *At what point does adding more context stop helping?*
I used 1, 3, 5, 10, and 20 as k. In this test, when k > 5, adding context no longer helps and instead starts to have the opposite effect.

### *When does too much context hurt (irrelevant information, confusion)?*
When K reaches 10 and 20, excessive context has a very serious negative impact. Even confusing 2 inches and 2 millimeters in the same sentence completely destroys the ability to reason.

### *How does k interact with chunk size?*
When K=20, assuming a chunk size of 512 characters, the model needs to process tens of thousands of characters at once. For small-parameter models, this places an excessive burden on the context window.
So, I believe that when using larger chunks (e.g., 1024), the K value should be decreased; otherwise, the model is prone to logical collapse, as seen with K=20. If smaller chunks (e.g., 128), the K value can be appropriately increased to piece together the answer like a jigsaw puzzle, without making the total prompt too long.

## Exercise 5: Handling Unanswerable Questions
### *Does the model admit it doesn't know?*
With the default prompt: It depends. For questions that are completely off-topic or have missing data, the model may sometimes admit that the context doesn't provide the information.
With the modified prompt: It performs very well. It will admit what it doesn't know based on the prompt.

### *Does it hallucinate plausible-sounding but wrong answers?*
Yes, especially when dealing with "false premises." These premises sometimes forcibly link irrelevant retrieved information with true information, fabricating a plausible but incorrect explanation.

### *Does retrieved context help or hurt? (Does irrelevant context encourage hallucination?)*
In some cases, the retrieved context actually hurts. This is because the default RAG prompts often tell the model to "answer the question based on the following context." This puts pressure on the model to provide a forced answer. So sometimes, in order to complete the task, the model will force a causal relationship between your false premises and irrelevant context, much like performing a "cloze test." This is a typical mechanism by which RAGs create serious illusions.

## Exercise 6: Query Phrasing Sensitivity
### *Which phrasings retrieve the best chunks?*
Based on my results, formal and keyword-only queries retrieved the best-fitting data blocks. In contrast, specific queries performed the worst. They had zero overlap with all other queries. For example, the query "When do I need to check the engine?" used the word "check," causing vector search to misinterpret it as "mechanical diagnostic test," thus recalling specific troubleshooting steps such as "turning the engine over" and observing the "muffler outlet," completely deviating from the intended meaning.

### *Do keyword-style queries work better or worse than natural questions?*
I believe keyword-only queries perform exceptionally well, outperforming most natural queries. The results show the highest overlap between "Keywords only" and "Formal" queries (sharing 3 chunks: 73, 54, 55). This demonstrates that even after removing all grammatical structures and retaining only the core content, it still accurately targets the key maintenance sections in the manual.
For natural queries, containing too many colloquial and ambiguous verbs, significantly hinder semantic matching.

### *What does this tell you about potential query rewriting strategies?*
The preceding analysis shows that raw input is highly unreliable; therefore, I believe a standardization mechanism is essential in RAGs. In practical applications, users are likely to ask questions casually or in informal question forms. Since technical manuals use highly formal and technical language, directly using the user's verbal language for vector retrieval can easily lead to errors due to "vocabulary mismatch."
I suggest adding an LLM rewriting step before the retrieval unit: regardless of the user's question, the large model should first standardize it.

## Exercise 7: Chunk Overlap Experiment
### *Does higher overlap improve retrieval of complete information?*
Moderate overlap significantly improves the retrieval of complete information across data block boundaries.
For example, when Overlap = 0: The model only retrieved steps 1 (removing the wall) and 2 (removing the splash guard and overflow pipe). Because the relevant paragraphs were cut off, the model even hallucinated at the end of the response, incorrectly including "cut through the top with a hacksaw" from the "Second Method" in the answer for the first method.
When Overlap = 64: The retrieval system successfully captured the cut-off subsequent steps. The model's response became accurate.
When Overlap = 256: The model also retrieved the complete steps.

### *What's the cost? (Index size, redundant information in context)*
The trade-off is that it results in highly redundant context input to the LLM.
The runtime logs show:
Overlap = 0: 1058 blocks
Overlap = 64: 1239 blocks
Overlap = 128: 1496 blocks
Overlap = 256: 2397 blocks.
This more than doubles the number of blocks, meaning it requires more than twice the memory, and the cost of API calls also doubles.

### *Is there a point of diminishing returns?*
I believe there is a clear point of diminishing marginal returns.
The results show that increasing from 0 to 64 yields the greatest benefit. However, increasing from 128 to 256 does not improve the quality of the generated answers, although the number of blocks increases significantly.

## Exercise 8: Chunk Size Experiment
### *How does chunk size affect retrieval precision (relevant vs. irrelevant content)?*
Smaller blocks result in higher keyword accuracy but lower semantic accuracy; larger blocks can accommodate more relevant background information but may introduce irrelevant noise.

### *How does it affect answer completeness?*
Too small a chunk will result in an incomplete answer, while larger chunks ensure the integrity of steps and logic.
Based on the results: 
128 characters: Extremely incomplete answer.
512 characters: Improved completeness.
2048 characters: Extremely complete answer.

### *Is there a sweet spot for your corpus?*
I believe there is a sweet spot for the Ford Model T repair manual, likely between 512 and 1024 characters. 128 characters is absolutely unusable, as it breaks up the sequential repair steps. 2048 characters performs best in preserving the complete steps, but it's prone to creating illusions. 512 characters provide sufficient, coherent context in most cases.

### *Does optimal size depend on the type of question?*
Yes. For fact-based questions, these questions don't require thousands of words of context. If the chunks are too large, lengthy explanations can actually interfere with the model's ability to find the specific content.
For process-based and general concepts questions, these questions often span several paragraphs. If the chunks are too small, the retrieval system will never be able to pass all the steps to the model simultaneously.

## Exercise 9: Retrieval Score Analysis
### *When is there a clear "winner" (large gap between #1 and #2)?*
When a question clearly corresponds to a specific process in the manual, a clear winner emerges. For example, Query 3 has a Top-1 score of 0.7116, far surpassing the second-place score of 0.6343, with a gap of 0.0773.

### *When are scores tightly clustered (ambiguous)?*
When multiple overlapping blocks exist, the scores cluster tightly. In Query 8, because the steering system is mentioned throughout the manual, its top 10 scores gradually drop from 0.53 to 0.42, with a very small gap between the first and second place scores of only 0.0074.

### *What score threshold would you use to filter out irrelevant results?*
Based on my results, I would set the threshold between 0.3 and 0.4.
This is because truly relevant queries (Q2, Q3, Q8) all have Top-1 scores above 0.50.
Completely irrelevant queries (Q9, Q10) have Top-1 scores between 0.06 and 0.20.
Some borderline issues (Q6 tire repair, Q7 wheelbase) have a maximum score of around 0.35.
### *How does score distribution correlate with answer quality?*
The top score determines whether the model will give false information: if the top-1 score is very low (<0.3), it means that the model is being fed useless information and the model is very likely to be delusional; if the score is very high (>0.5), the answer is usually accurate and based on facts.
The gap determines the focus of the model's answer: If there is a clear large gap, the model will directly provide extremely precise steps based on that section; if the high scores are closely clustered , it means that the relevant information is scattered in several blocks, and the model needs to piece these blocks together to answer the question.

## Exercise 10: Prompt Template Variations
### *Which prompt produces the most accurate answers?*
The results clearly show that strict grounding produces the most accurate answers. Other hints produce answers that are more or less biased.

### *Which produces the most useful answers?*
In terms of practicality, structured output is most useful for "answerable questions"; however, if absolute security is the goal, strict grounding remains the better choice.

### *Is there a trade-off between strict grounding and helpfulness?*
I believe there's a significant trade-off: pursuing helpfulness means the model will be overly eager to help the user, even if it doesn't know the correct answer, creating an illusion. On the other hand, pursuing strict grounding comes at the cost of absolute safety—it never fabricates answers. Even if the question is only slightly off-limits, it will refuse to answer, which diminishes the user experience.

## Exercise 11: Cross-Document Synthesis
### *Can the model successfully combine information from multiple chunks?*
Yes, the model has good information extraction and synthesis capabilities, but these capabilities are limited by the model's parameter size.

### *Does it miss information that wasn't retrieved?*
Retrieving more blocks provides a more comprehensive information base, because a K value that is too small will inevitably lead to information omissions; however, an excessively large K value will cause the small model to become logically chaotic and unable to process all the information, resulting in missing information.

### *Does contradictory information in different chunks cause problems?*
This can cause problems, especially when the system extracts too many "incompletely matching" edge blocks, leading to logical inconsistencies and noise pollution.
In my results, the system forcibly pulled a large amount of irrelevant information to fill in the gaps in order to make up 10 blocks.


## Group
Chenxu Li-jnr2jp,
Wenhao Xu-wx8mcm

## Project Topics
* [**Topic 1: Running an LLM**](../../tree/Topic-1-Running-an-LLM)
* [**Topic 2: Agent Orchestration Frameworks**](../../tree/Topic-2-Agent-Orchestration-Frameworks)
* [**Topic 3: Agent Tool Use**](../../tree/Topic-3-Agent-Tool-Use)
* [**Topic 4: Exploring Tools**](../../tree/Topic-4-Exploring-Tools)
* [**Topic 5: RAG**](../../tree/Topic-5-RAG)
* [**Topic 6: VLM**](../../tree/Topic-6-VLM)
* [**Topic 7: MCP&A2A**](../../tree/Topic-7-MCP-and-A2A)
