"""FAISS index + JSON metadata; embeddings via configured LLM provider (Ollama default)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from app.config import EMBEDDING_MODEL
from app.core.llm import get_provider

logger = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vectors)
    return vectors


def _default_embed(texts: list[str]) -> list[list[float]]:
    return get_provider().embed(model=EMBEDDING_MODEL, texts=texts)


class CodexvidVectorStore:
    """Disk-backed FAISS (inner product on L2-normalized vectors ≈ cosine)."""

    def __init__(self, session_dir: Path, dim: int, index: faiss.Index, meta: list[dict]):
        self.session_dir = Path(session_dir)
        self.dim = dim
        self.index = index
        self.meta = meta

    @classmethod
    def build(
        cls,
        chunks: list[dict],
        session_dir: Path,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> CodexvidVectorStore:
        if not chunks:
            raise ValueError("chunks must be non-empty")
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        embed_fn = embed_fn or _default_embed
        texts = [c["text"] for c in chunks]
        embeddings = embed_fn(texts)
        if not embeddings or len(embeddings) != len(chunks):
            raise RuntimeError("embedding failed or length mismatch")
        dim = len(embeddings[0])
        arr = np.array(embeddings, dtype=np.float32)
        _normalize(arr)
        index = faiss.IndexFlatIP(dim)
        index.add(arr)
        meta = []
        for c in chunks:
            st = float(c.get("start_time", c.get("start", 0.0)))
            en = float(c.get("end_time", c.get("end", 0.0)))
            meta.append(
                {
                    "text": c["text"],
                    "start_time": st,
                    "end_time": en,
                    "start": st,
                    "end": en,
                }
            )
        return cls(session_dir, dim, index, meta)

    @classmethod
    def build_empty(cls, session_dir: Path, embed_fn: Callable[[list[str]], list[list[float]]] | None = None) -> CodexvidVectorStore:
        """Zero-vector index so :meth:`save` creates ``faiss.index`` when there are no chunks."""
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        embed_fn = embed_fn or _default_embed
        probe = embed_fn(["."])
        if not probe or not probe[0]:
            raise RuntimeError("embedding probe failed for empty index")
        dim = len(probe[0])
        index = faiss.IndexFlatIP(dim)
        return cls(session_dir, dim, index, [])

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        embed_fn = _default_embed
        qv = np.array(embed_fn([query.strip()]), dtype=np.float32)
        if qv.size == 0:
            return []
        _normalize(qv)
        k = min(k, self.index.ntotal)
        if k <= 0:
            return []
        scores, idxs = self.index.search(qv, k)
        out: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            row = dict(self.meta[idx])
            row["score"] = float(score)
            out.append(row)
        return out

    def save(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.session_dir / "faiss.index"))
        payload = {"dim": self.dim, "meta": self.meta}
        (self.session_dir / "faiss_meta.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, session_dir: Path) -> CodexvidVectorStore:
        session_dir = Path(session_dir)
        index = faiss.read_index(str(session_dir / "faiss.index"))
        data = json.loads((session_dir / "faiss_meta.json").read_text(encoding="utf-8"))
        dim = int(data["dim"])
        meta = data["meta"]
        return cls(session_dir, dim, index, meta)


def build_vector_store(
    chunks: list[dict],
    session_dir: str | Path,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> CodexvidVectorStore:
    """Index chunks under ``session_dir`` (caller should save)."""
    return CodexvidVectorStore.build(chunks, Path(session_dir), embed_fn=embed_fn)
