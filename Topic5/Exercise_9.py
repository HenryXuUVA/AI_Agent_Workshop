# Exercise 9: Retrieval Score Analysis

# 10 queries
queries_for_analysis = [
    "What is the correct spark plug gap?",                 # 1. Specific
    "How do I adjust the carburetor?",                    # 2. Specific process
    "What are the steps to remove the top tank top?",          # 3. Long process
    "What is the recommended maintenance schedule?",            # 4. General
    "What oil should I use in the engine?",                # 5. Recommendation
    "How to repair a flat tire?",                      # 6. Common but maybe missing
    "What is the wheelbase of the Model T?",                # 7. Specific fact
    "Tell me about the steering gear.",                   # 8. Broad topic
    "What is the capital of France?",                    # 9. Completely off-topic
    "Why does the manual recommend synthetic oil?"             # 10. False premise
]

if index.ntotal > 0:
    print("="*80)
    print("PART 1: Score Distribution Analysis (Top 10)")
    print("="*80)

    for i, q in enumerate(queries_for_analysis, 1):
        print(f"\n[Query {i}] {q}")
        results = retrieve(q, top_k=10)

        scores = [round(score, 4) for chunk, score in results]
        gap = scores[0] - scores[1] if len(scores) > 1 else 0

        print(f"  Scores: {scores}")
        print(f"  Gap (#1 - #2): {gap:.4f} | Top Score: {scores[0]:.4f}")

    print("\n\n" + "="*80)
    print("PART 2: Experiment - Implementing a Score Threshold (> 0.5)")
    print("="*80)

    def retrieve_with_threshold(query, top_k=5, threshold=0.5):
        results = retrieve(query, top_k=top_k)
        filtered_results = [(chunk, score) for chunk, score in results if score > threshold]
        return filtered_results

    # Test the threshold
    test_query = "What is the capital of France?"
    print(f"\nTesting Threshold on: '{test_query}'")

    # 1. Without Threshold
    raw_results = retrieve(test_query, top_k=3)
    print(f"  Chunks retrieved WITHOUT threshold: {len(raw_results)}")

    # 2. With Threshold > 0.5
    filtered_results = retrieve_with_threshold(test_query, top_k=3, threshold=0.5)
    print(f"  Chunks retrieved WITH threshold > 0.5: {len(filtered_results)}")

    if len(filtered_results) == 0:
        print("  -> System Action: Prevent LLM call. Return 'No relevant documents found.' directly.")
else:
    print("Please complete the pipeline setup first.")