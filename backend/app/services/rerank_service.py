from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates):
    pairs = [(query, c["title"]) for c in candidates]
    scores = reranker.predict(pairs)

    for i in range(len(candidates)):
        candidates[i]["rerank_score"] = scores[i]

    candidates = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return candidates