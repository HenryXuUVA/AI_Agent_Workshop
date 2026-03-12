# Exercise 10: Prompt Template Variations


# 5 prompt
prompts = {
    "Minimal": """
{context}

{question}
""",

    "Strict Grounding": """
Answer ONLY based on the context. If the answer isn't there, say 'I cannot answer this from the available documents.'
CONTEXT:
{context}

QUESTION: {question}
ANSWER:""",

    "Encouraging Citation": """
Answer the question based on the context. Quote the exact passages that support your answer using quotation marks.
CONTEXT:
{context}

QUESTION: {question}
ANSWER:""",

    "Permissive": """
Use the context to help answer the question, but you may also use your own general knowledge if the context is insufficient.
CONTEXT:
{context}

QUESTION: {question}
ANSWER:""",

    "Structured Output": """
First list relevant facts from the context in bullet points, then synthesize your final answer.
CONTEXT:
{context}

QUESTION: {question}

FACTS AND SYNTHESIS:"""
}

# 5 test queries
test_queries = [
    "What is the correct spark plug gap for a Model T Ford?",
    "What are the steps to remove the top tank top using the first method?",
    "What is the recommended maintenance schedule?",
    "Why does the manual recommend synthetic oil?",
    "How to repair a flat tire?"
]

if index.ntotal > 0:
    print("="*80)
    print("Exercise 10: Testing Prompt Variations")
    print("="*80)

    for template_name, prompt_text in prompts.items():
        print(f"\n\n{'#'*80}")
        print(f"PROMPT STYLE: {template_name.upper()}")
        print(f"{'#'*80}")

        for i, q in enumerate(test_queries, 1):
            print(f"\n[Query {i}] {q}")
            print("-" * 40)

            # top_k=3
            answer = rag_query(q, top_k=3, show_context=False, prompt_template=prompt_text)


            print(answer)
else:
    print("Please complete the pipeline setup first.")