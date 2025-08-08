import numpy as np
from typing import List, Dict

class MemoryStore:
    """Simple in-memory vector store for text segments."""
    def __init__(self) -> None:
        self.segments: List[Dict[str, np.ndarray]] = []

    def add(self, text: str, embedding: List[float]) -> None:
        self.segments.append({"text": text, "embedding": np.array(embedding, dtype=float)})

    def search(self, embedding: List[float], top_k: int = 3) -> List[str]:
        if not self.segments:
            return []
        query = np.array(embedding, dtype=float)
        # cosine similarity
        scores = [
            float(np.dot(seg["embedding"], query) /
                  (np.linalg.norm(seg["embedding"]) * np.linalg.norm(query)))
            for seg in self.segments
        ]
        indices = np.argsort(scores)[::-1][:top_k]
        return [self.segments[i]["text"] for i in indices]
