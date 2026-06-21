"""
rag.py — Core RAG (Retrieval-Augmented Generation) logic.

Mental model in plain English:

  1. INGEST: Take a document, break it into small overlapping chunks of text.
  2. EMBED: Convert each chunk into a vector (list of numbers) capturing meaning.
     Similar meaning = similar vector.
  3. STORE: Keep those vectors in a FAISS index for fast similarity search.
     Persist to disk so documents survive server restarts.
  4. RETRIEVE: Embed the question, find the top-k most similar chunks.
  5. GENERATE: Send those chunks + question to an LLM: "answer ONLY from this."

RAG beats pasting the whole doc into a prompt because:
  - Docs can exceed the context window
  - Retrieval is cheaper (4 chunks vs. whole doc every query)
  - Answers are auditable — you see exactly what produced the response
"""

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import faiss
from openai import OpenAI
from pypdf import PdfReader

_client = None

def get_client():
    """Lazy-initialize OpenAI client on first use.
    This allows the server to start even if OPENAI_API_KEY is not yet set.
    """
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from environment
    return _client

EMBEDDING_MODEL   = "text-embedding-3-small"   # 1536-dim, cheap, good enough
CHAT_MODEL        = "gpt-4o-mini"               # cheap + fast; swap for gpt-4o for better answers
CHUNK_SIZE        = 800      # characters per chunk
CHUNK_OVERLAP     = 150      # overlap so sentences aren't cut across chunk boundaries
TOP_K             = 4        # chunks retrieved per question
PERSIST_DIR       = Path("data")   # where FAISS index + metadata are saved to disk

assert CHUNK_OVERLAP < CHUNK_SIZE, "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"


@dataclass
class Chunk:
    text: str
    source: str       # original filename
    chunk_id: int


@dataclass
class DocumentStore:
    """
    Vector store backed by FAISS with disk persistence.

    On startup: loads index + chunks from disk if they exist.
    On add: saves index + chunks to disk after each ingest.

    This means documents survive server restarts — a critical requirement
    for any production deployment. The in-memory-only version would wipe
    everything every time Railway/Render restarts the container.

    Swap FAISS for Chroma/Pinecone/pgvector for multi-tenant scale.
    The retrieval interface (search method signature) stays identical.
    """
    dim: int = 1536
    index: faiss.IndexFlatIP = field(default=None)
    chunks: list = field(default_factory=list)

    def __post_init__(self):
        PERSIST_DIR.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        self._try_load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _index_path(self) -> Path:
        return PERSIST_DIR / "faiss.index"

    def _meta_path(self) -> Path:
        return PERSIST_DIR / "chunks.json"

    def _try_load(self):
        """Load index + chunks from disk if saved files exist."""
        if self._index_path().exists() and self._meta_path().exists():
            try:
                self.index = faiss.read_index(str(self._index_path()))
                raw = json.loads(self._meta_path().read_text())
                self.chunks = [Chunk(**c) for c in raw]
            except Exception:
                # Corrupt files — start fresh
                self.index = faiss.IndexFlatIP(self.dim)
                self.chunks = []

    def _save(self):
        """Persist index + chunk metadata to disk."""
        faiss.write_index(self.index, str(self._index_path()))
        self._meta_path().write_text(
            json.dumps([{"text": c.text, "source": c.source, "chunk_id": c.chunk_id}
                        for c in self.chunks])
        )

    # ── Core operations ───────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, chunks: list[Chunk]):
        with self._lock:
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            self.chunks.extend(chunks)
            self._save()

    def search(self, query_vector: np.ndarray, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self.index.ntotal == 0:
                return []
            faiss.normalize_L2(query_vector)
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self.chunks[idx], float(score)))
            return results

    def list_sources(self) -> list[str]:
        return sorted(set(c.source for c in self.chunks))

    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def reset(self):
        """Clear in-memory store and delete persisted files."""
        with self._lock:
            self.index = faiss.IndexFlatIP(self.dim)
            self.chunks = []
            for p in [self._index_path(), self._meta_path()]:
                if p.exists():
                    p.unlink()


# Single global store (one document collection per server instance).
STORE = DocumentStore()


# ── Text extraction ────────────────────────────────────────────────────────

def extract_text(file_path: str, filename: str) -> str:
    """Pull raw text from a PDF or plain text file."""
    if filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise ValueError(
                f"Could not read {filename} as a PDF ({e}). "
                "Is it password-protected or corrupted?"
            )
    else:
        return Path(file_path).read_text(errors="ignore")


# ── Chunking ───────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[Chunk]:
    """
    Split text into overlapping fixed-size chunks.

    Why overlap? If a chunk boundary falls mid-sentence, the overlap
    ensures that sentence still appears whole in the next chunk, so
    retrieval doesn't miss it.

    Improvement over this: semantic chunking (split on embedding distance
    or paragraph boundaries). Noted as a known limitation in the README.
    """
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_id=chunk_id))
            chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Embeddings ─────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Call OpenAI's embedding model on a batch of strings.

    Batches into groups of 500 to stay under OpenAI's 2048-input-per-request
    limit. Without batching, large documents (500+ pages) would throw a
    cryptic API error mid-ingest.
    """
    BATCH_SIZE = 500
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_vectors.extend([d.embedding for d in response.data])
    return np.array(all_vectors, dtype="float32")


# ── Pipeline ───────────────────────────────────────────────────────────────

def ingest_document(file_path: str, filename: str) -> int:
    """Full ingest pipeline for one uploaded file. Returns number of chunks added."""
    text = extract_text(file_path, filename)
    if not text.strip():
        raise ValueError(f"No extractable text found in {filename}")

    chunks = chunk_text(text, filename)
    vectors = embed_texts([c.text for c in chunks])
    STORE.add(vectors, chunks)
    return len(chunks)


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    """
    Full RAG query pipeline with optional conversation history:
      embed question → retrieve top-k chunks → build prompt → call LLM

    history: list of {role: "user"|"assistant", content: str} from previous turns.
    Passing history enables multi-turn conversation — the model can reference
    earlier answers when answering follow-up questions.

    Returns answer + source chunks used, so the UI can show where answers came from.
    """
    if STORE.is_empty():
        return {
            "answer": "No documents uploaded yet. Upload a PDF or text file first.",
            "sources": [],
        }

    query_vector = embed_texts([question])
    results = STORE.search(query_vector, k=TOP_K)

    context_blocks = []
    sources = []
    for chunk, score in results:
        context_blocks.append(
            f"[Source: {chunk.source}, chunk {chunk.chunk_id}]\n{chunk.text}"
        )
        sources.append({
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "score": round(score, 3),
            "preview": chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
        })

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the provided context. "
        "If the context doesn't contain the answer, say so clearly instead of guessing. "
        "Cite which source/chunk you used when relevant."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Inject conversation history so follow-ups have context
    if history:
        messages.extend(history[-6:])  # last 3 turns (6 messages) to avoid token bloat

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above."
    })

    response = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }


def reset_store():
    STORE.reset()
