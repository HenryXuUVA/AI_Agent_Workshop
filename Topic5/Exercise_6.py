# Exercise 6: Query Phrasing Sensitivity
phrasings = {
    "Formal": "What is the recommended maintenance schedule for the engine?",
    "Casual": "How often should I service the engine?",
    "Keywords only": "engine maintenance intervals",
    "Question form": "When do I need to check the engine?",
    "Indirect": "Preventive maintenance requirements"
}

if index.ntotal > 0:
    print("="*60)
    print("Exercise 6: Query Phrasing Sensitivity")
    print("="*60)

    retrieved_sets = {}

    # Retrieve and record top 5 chunks for each phrasing
    for style, query in phrasings.items():
        print(f"\n[{style}] Query: {query}")
        print("-" * 60)

        results = retrieve(query, top_k=5)
        chunk_indices = set()

        for i, (chunk, score) in enumerate(results, 1):
            chunk_indices.add(chunk.chunk_index)
            clean_text = chunk.text[:80].replace('\n', ' ')
            print(f"Top {i} | Score: {score:.4f} | Chunk ID: {chunk.chunk_index:^4} | Text: {clean_text}...")

        retrieved_sets[style] = chunk_indices

    # Compare
    print("\n" + "="*60)
    print("Overlap Analysis (Shared Chunk IDs)")
    print("="*60)

    styles = list(phrasings.keys())
    for i in range(len(styles)):
        for j in range(i+1, len(styles)):
            s1, s2 = styles[i], styles[j]
            overlap = retrieved_sets[s1].intersection(retrieved_sets[s2])
            print(f"{s1} vs {s2}: {len(overlap)} shared chunks -> {list(overlap)}")
else:
    print("Please complete the pipeline setup first.")