import numpy as np
from typing import List, Dict, Any, Optional

class MemoryStore:
    """Simple in-memory vector store for text segments."""
    def __init__(self) -> None:
        self.segments: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None

    def add(self, text: str, embedding: List[float], kind: Optional[str] = None) -> None:
        """Add a text segment and its embedding to the store.

        Args:
            text: The text to store.
            embedding: Vector embedding for the text.
            kind: Optional metadata to distinguish entry type
                (e.g., "prompt" or "response").
        """

        if self._dim is None:
            self._dim = len(embedding)
        elif len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dim}, got {len(embedding)}"
            )

        self.segments.append(
            {
                "text": text,
                "embedding": np.array(embedding, dtype=float),
                "kind": kind,
            }
        )

    def search(
        self,
        embedding: List[float],
        top_k: int = 3,
        kind: Optional[str] = None,
    ) -> List[str]:
        """Search for similar text segments.

        Args:
            embedding: Query embedding.
            top_k: Number of results to return.
            kind: Optional metadata filter; if provided, only
                segments with matching ``kind`` are considered.
        """

        if not self.segments:
            return []

        if self._dim is not None and len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dim}, got {len(embedding)}"
            )

        query = np.array(embedding, dtype=float)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        scored = []
        for seg in self.segments:
            if kind is not None and seg.get("kind") != kind:
                continue
            seg_emb = seg["embedding"]
            seg_norm = np.linalg.norm(seg_emb)
            if seg_norm == 0:
                continue
            score = float(np.dot(seg_emb, query) / (seg_norm * query_norm))
            scored.append((score, seg["text"]))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]
