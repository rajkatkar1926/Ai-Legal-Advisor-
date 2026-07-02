from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import KNOWLEDGE_BASE

LOW_CONFIDENCE_THRESHOLD = 0.1

_corpus = [
    f"{entry['topic']} {entry['law']} {entry['summary']}" for entry in KNOWLEDGE_BASE
]
_vectorizer = TfidfVectorizer(stop_words="english")
_kb_matrix = _vectorizer.fit_transform(_corpus)


def retrieve_top_k(query, k=2):
    """Return the top-k KB entries most similar to the query, each with a
    'score' field and a 'low_confidence' flag if even the best match is weak."""
    query_vector = _vectorizer.transform([query])
    scores = cosine_similarity(query_vector, _kb_matrix)[0]

    ranked_indices = scores.argsort()[::-1][:k]
    best_score = scores[ranked_indices[0]]
    low_confidence = best_score < LOW_CONFIDENCE_THRESHOLD

    results = []
    for idx in ranked_indices:
        entry = dict(KNOWLEDGE_BASE[idx])
        entry["score"] = float(scores[idx])
        entry["low_confidence"] = low_confidence
        results.append(entry)
    return results


if __name__ == "__main__":
    test_queries = [
        "what is the POSH Act",
        "what can I do if my landlord harasses me",
        "my husband hits me",
        "someone is sharing my morphed photos online",
        "what's the weather today",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        for r in retrieve_top_k(q):
            print(f"  [{r['score']:.3f}] {r['topic']} — {r['law']}"
                  f"{' (low confidence)' if r['low_confidence'] else ''}")
