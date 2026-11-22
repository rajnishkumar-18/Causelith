import os
import re
import json
import glob
from typing import List, Dict, Any
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PDF_DIR = os.path.join("data", "pdfs")
INDEX_PATH = os.path.join("data", "vector_index.npz")
CHUNKS_PATH = os.path.join("data", "chunks.json")


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer that preserves policy/incident IDs like POL-APAC-402."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def _get_embedder() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip().strip("'").strip('"')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the .env file.")

    candidate_models = [
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "models/text-embedding-004",
    ]
    last_error = None
    for model_name in candidate_models:
        try:
            embedder = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
            # Lightweight verification call
            embedder.embed_query("healthcheck")
            print(f"[OK] Embedding model active: {model_name}")
            return embedder
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not initialize Gemini embeddings. Last error: {last_error}")


def _chunk_text(text: str, max_chars: int = 750, overlap: int = 120) -> List[str]:
    """Splits document text into overlapping chunks while respecting paragraph breaks."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            last_period = cleaned.rfind(". ", start, end)
            if last_period > start + int(max_chars * 0.5):
                end = last_period + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _extract_doc_id(text: str, fallback: str) -> str:
    match = re.search(r"Document ID:\s*([A-Za-z0-9\-]+)", text)
    return match.group(1) if match else fallback


class HybridRAGEngine:
    """Hybrid Dense (Gemini Embeddings + Cosine) + Sparse (BM25) Retriever with RRF."""

    def __init__(self):
        self.embedder: GoogleGenerativeAIEmbeddings | None = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = _get_embedder()

    def build_index(self, pdf_dir: str = PDF_DIR) -> int:
        """Parses all PDFs, creates chunks with metadata, embeds them, and saves the index."""
        pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'. Run Phase 2 first.")

        self._ensure_embedder()
        all_chunks: List[Dict[str, Any]] = []

        for path in pdf_files:
            filename = os.path.basename(path)
            reader = PdfReader(path)
            full_text = "\n".join((page.extract_text() or "") for page in reader.pages)
            doc_id = _extract_doc_id(full_text, fallback=os.path.splitext(filename)[0])

            for idx, chunk_text in enumerate(_chunk_text(full_text)):
                all_chunks.append(
                    {
                        "chunk_id": f"{doc_id}#c{idx + 1}",
                        "doc_id": doc_id,
                        "source_file": filename,
                        "text": chunk_text,
                    }
                )

        texts = [c["text"] for c in all_chunks]
        print(f"Embedding {len(texts)} document chunks from {len(pdf_files)} PDFs...")
        raw_vectors = self.embedder.embed_documents(texts)
        matrix = np.array(raw_vectors, dtype=np.float32)

        # L2 normalize so dot product equals cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        np.savez_compressed(INDEX_PATH, embeddings=matrix)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)

        self.chunks = all_chunks
        self.embeddings = matrix
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

        print(f"[OK] Hybrid index built and saved ({len(self.chunks)} chunks indexed).")
        return len(self.chunks)

    def load_or_build(self):
        """Loads persisted index from disk, or builds it automatically if missing."""
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            data = np.load(INDEX_PATH)
            self.embeddings = data["embeddings"]
            self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])
            self._ensure_embedder()
        else:
            self.build_index()

    def search(self, query: str, top_k: int = 4, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Runs Dense + BM25 search and fuses rankings via Reciprocal Rank Fusion (RRF)."""
        if self.embeddings is None or self.bm25 is None:
            self.load_or_build()

        self._ensure_embedder()

        # 1. Dense semantic ranking
        q_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        dense_scores = self.embeddings @ q_vec
        dense_order = np.argsort(-dense_scores)

        # 2. Sparse BM25 keyword ranking
        bm25_scores = np.array(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)
        bm25_order = np.argsort(-bm25_scores)

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(dense_order):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_order):
            if bm25_scores[idx] > 0:
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)

        ranked_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            item = dict(self.chunks[idx])
            item["rrf_score"] = round(float(rrf_scores[idx]), 5)
            item["semantic_score"] = round(float(dense_scores[idx]), 4)
            results.append(item)
        return results


if __name__ == "__main__":
    engine = HybridRAGEngine()
    engine.build_index()

    test_queries = [
        "Why did APAC Enterprise conversion drop and pricing approvals get delayed?",
        "Why did Europe Pro tier discounts spike above 25%?",
        "Why did North America Mid-Market customer churn increase in November?",
    ]

    print("\n--- Running Hybrid Retrieval Self-Test ---")
    for q in test_queries:
        print(f"\nQuery: {q}")
        hits = engine.search(q, top_k=2)
        for h in hits:
            print(f"  -> [{h['doc_id']}] {h['source_file']} (semantic={h['semantic_score']}, rrf={h['rrf_score']})")

    print("\n=== PHASE 3 COMPLETE! Hybrid RAG Engine is ready. ===")