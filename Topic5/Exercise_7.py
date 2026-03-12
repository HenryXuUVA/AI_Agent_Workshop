# Exercise 7: Chunk Overlap Experiment

import time

overlaps_to_test = [0, 64, 128, 256]
fixed_chunk_size = 512
cross_boundary_query = "What are the steps to remove the top tank top using the first method?"

print("="*60)
print("Exercise 7: Testing Chunk Overlap (Size=512)")
print("="*60)

for overlap in overlaps_to_test:
    print(f"\n\n{'*'*60}")
    print(f"Rebuilding Index | Size: {fixed_chunk_size} | Overlap: {overlap}")
    print(f"{'*'*60}")

    start_time = time.time()

    rebuild_pipeline(chunk_size=fixed_chunk_size, chunk_overlap=overlap)

    rebuild_time = time.time() - start_time
    print(f"Time taken: {rebuild_time:.2f}s | Total Chunks: {len(all_chunks)}")

    #
    print(f"\nQuery: {cross_boundary_query}")
    print("-" * 40)

    #
    answer = rag_query(cross_boundary_query, top_k=3, show_context=True)

    print("\nAnswer:")
    print(answer)