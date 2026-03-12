# Exercise 8: Chunk Size Experiment

import time


chunk_sizes_to_test = [128, 512, 2048]
fixed_overlap = 20

test_queries = [
    "What is the correct spark plug gap for a Model T Ford?",            # Specific fact
    "How do I adjust the carburetor?",                                   # Short process
    "What are the steps to remove the top tank top using the first method?", # Long process
    "What oil should I use in the engine?",                              # Fact/Recommendation
    "What is the recommended maintenance schedule?"                      # General concept
]

print("="*60)
print("Exercise 8: Testing Chunk Size (128, 512, 2048)")
print("="*60)

for size in chunk_sizes_to_test:
    print(f"\n\n{'*'*60}")
    print(f"Rebuilding Index | Chunk Size: {size} | Overlap: {fixed_overlap}")
    print(f"{'*'*60}")

    start_time = time.time()
    # Rebuild index with the current chunk size
    rebuild_pipeline(chunk_size=size, chunk_overlap=fixed_overlap)
    rebuild_time = time.time() - start_time

    print(f"Time taken: {rebuild_time:.2f}s | Total Chunks: {len(all_chunks)}")

    # Run the 5 queries
    for i, q in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {q} ---")
        # Set show_context=False to keep output clean, but you can change it if needed
        answer = rag_query(q, top_k=3, show_context=False)
        # Truncate answer for cleaner output display
        print(f"Answer: {answer[:300]}...\n")