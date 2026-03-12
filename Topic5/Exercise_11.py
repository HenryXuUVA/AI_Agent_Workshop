# Exercise 11: Cross-Document/Cross-Chunk Synthesis

synthesis_queries = [
    "What are ALL the maintenance tasks, oil changes, or checks I need to perform regularly?", # Task synthesis
    "What specific tools or equipment are mentioned for repairing the radiator and tank?",    # Tool synthesis
    "Summarize any safety warnings, care instructions, or precautions mentioned."          # Warning synthesis
]

k_values = [3, 5, 10]

if index.ntotal > 0:
    print("="*80)
    print("Exercise 11: Cross-Chunk Synthesis Experiment")
    print("="*80)

    for i, q in enumerate(synthesis_queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"SYNTHESIS QUERY {i}: {q}")
        print(f"{'#'*80}")

        for k in k_values:
            print(f"\n--- Testing with top_k = {k} ---")
            # prompt
            prompt = """Synthesize a comprehensive answer based ONLY on the provided context.
List all relevant points you can find. If pieces of information are in different paragraphs, combine them logically.
CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""

            # Show context is False to keep output clean, but you can turn it on if you want to see the raw chunks
            answer = rag_query(q, top_k=k, show_context=False, prompt_template=prompt)
            print(answer)
            print("-" * 60)
else:
    print("Please complete the pipeline setup first.")