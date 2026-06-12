import re
import nltk

for _pkg in ("punkt_tab", "punkt"):
    try:
        nltk.download(_pkg, quiet=True, raise_on_error=True)
        break
    except Exception:
        pass


def clean_text(text: str) -> str:
    """Strip page numbers, decorative lines, and normalize whitespace."""
    text = re.sub(r'(?m)^\s*(?:Page\s+\d+(?:\s+of\s+\d+)?|\d+)\s*$', '', text)
    text = re.sub(r'[-_]{4,}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def smart_chunk(pages: list[dict], target_words: int = 300, overlap_words: int = 50) -> list[dict]:
    """Sentence-aware chunking that avoids cutting mid-sentence."""
    chunks = []
    for page in pages:
        cleaned = clean_text(page["text"])
        sentences = nltk.sent_tokenize(cleaned)
        current_words: list[str] = []
        chunk_index = 0

        for sentence in sentences:
            sent_words = sentence.split()
            if not sent_words:
                continue
            if current_words and len(current_words) + len(sent_words) > target_words:
                chunks.append({
                    "page": page["page"],
                    "text": " ".join(current_words),
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
                overlap_start = max(0, len(current_words) - overlap_words)
                current_words = current_words[overlap_start:] + sent_words
            else:
                current_words.extend(sent_words)

        if current_words:
            chunks.append({
                "page": page["page"],
                "text": " ".join(current_words),
                "chunk_index": chunk_index,
            })

    return chunks


def embed_and_store(doc_id: str, chunks: list[dict], collection) -> None:
    """Batch embed chunks and store in a ChromaDB collection with metadata."""
    from rag.retriever import get_embedding_model
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "page": c["page"],
            "chunk_index": c.get("chunk_index", i),
            "word_count": len(c["text"].split()),
        }
        for i, c in enumerate(chunks)
    ]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
