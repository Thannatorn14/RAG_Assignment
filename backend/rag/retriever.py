import re
from sentence_transformers import SentenceTransformer

_model = None

# Triggers that suggest the query spans multiple distinct information needs
_COMPOUND_SIGNALS = ("and", "also", "as well as", "?")


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _is_compound(query: str) -> bool:
    q = query.lower()
    return (
        q.count("?") > 1
        or any(
            len(re.findall(r'\b' + re.escape(kw) + r'\b', q)) > 0
            for kw in ("and", "also", "as well as")
        )
    )


def _split_sub_queries(query: str) -> list[str]:
    """Split a compound query into individual sub-queries."""
    # Split on sentence boundaries, then on coordinating conjunctions
    parts = re.split(r'(?<=[.?!])\s+', query.strip())
    sub_queries = []
    for part in parts:
        # Further split on conjunctions; "and also" must precede "and" in the alternation
        segments = re.split(r'\s+(?:and also|and|also|as well as)\s+', part, flags=re.IGNORECASE)
        sub_queries.extend(s.strip() for s in segments if s.strip())
    # Deduplicate while preserving order, drop single-word fragments
    seen = set()
    result = []
    for sq in sub_queries:
        if sq not in seen and len(sq.split()) >= 2:
            seen.add(sq)
            result.append(sq)
    return result or [query]


class VectorStore:
    """Wraps a ChromaDB collection for semantic search."""

    def __init__(self, collection):
        self.collection = collection

    def _single_search(self, query: str, top_k: int) -> list[dict]:
        model = get_embedding_model()
        q_emb = model.encode([query], convert_to_numpy=True)
        n = min(top_k, self.collection.count())
        if n == 0:
            return []
        results = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=n,
        )
        output = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # cosine space: distance = 1 - cosine_similarity → invert for score
            score = 1.0 - distance
            output.append({
                "text": doc,
                "page": meta.get("page", 0),
                "score": float(score),
            })
        return output

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not _is_compound(query):
            return self._single_search(query, top_k)

        sub_queries = _split_sub_queries(query)

        # Collect results from each sub-query (top_k=2 per sub-query)
        seen_text: dict[str, dict] = {}
        for sq in sub_queries:
            for hit in self._single_search(sq, top_k=2):
                text = hit["text"]
                # Keep the highest score if the same chunk appears in multiple sub-queries
                if text not in seen_text or hit["score"] > seen_text[text]["score"]:
                    seen_text[text] = hit

        merged = sorted(seen_text.values(), key=lambda h: h["score"], reverse=True)
        return merged[:top_k]
