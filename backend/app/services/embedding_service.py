"""
Embedding Service
Handles text-to-vector embeddings using sentence-transformers
Used for semantic similarity search in memory and context matching

"""

from typing import Iterable, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Use GPU if available, otherwise CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize embedding model (all-MiniLM-L6-v2: fast, 384-dim vectors)
_model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device=DEVICE,
)


def embed(texts: Iterable[str] | str, normalize: bool = True) -> list[list[float]]:
    """
    Convert text strings to embedding vectors.

    Accepts a single string or an iterable of strings. A bare string MUST
    be wrapped here: list("abc") explodes into per-character items, which
    silently embeds single letters instead of the sentence.

    """
    if isinstance(texts, str):
        texts = [texts]
    embeddings = _model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    Returns value between -1 and 1 (higher = more similar)

    """
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
