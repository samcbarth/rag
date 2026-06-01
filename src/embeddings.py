"""
Embedding services — convert text to dense float vectors.

Two implementations are provided:

  HashingEmbeddingService  (default)
      Offline, no downloads. Uses sklearn's HashingVectorizer with character
      and word n-grams. Captures keyword / morphological similarity.
      Good enough for demos and environments without HuggingFace access.

  SentenceTransformerEmbeddingService
      Requires network access on first use to download the model weights.
      Produces semantically-aware embeddings — similar *meanings* score
      high even when wordings differ. Use this in production.
"""

from __future__ import annotations
from typing import Union
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class HashingEmbeddingService:
    """
    Offline dense embeddings via the hashing trick.

    Uses unigram + bigram token features mapped to a fixed-size vector
    with L2 normalisation. No fitting step, no downloads, deterministic.
    """

    def __init__(self, n_features: int = 512):
        self.dimension = n_features
        self._vec = HashingVectorizer(
            n_features=n_features,
            ngram_range=(1, 2),
            analyzer="word",
            norm="l2",
            alternate_sign=False,
        )

    def embed(self, text: Union[str, list[str]]) -> list[list[float]]:
        single = isinstance(text, str)
        inputs = [text] if single else text
        sparse = self._vec.transform(inputs)
        return sparse.toarray().tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed(text)[0]


class SentenceTransformerEmbeddingService:
    """
    Neural embeddings via sentence-transformers (requires HuggingFace access
    on first use to download model weights; subsequent runs use local cache).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, strong semantic quality

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: Union[str, list[str]]) -> list[list[float]]:
        single = isinstance(text, str)
        inputs = [text] if single else text
        vecs = self._model.encode(inputs, normalize_embeddings=True)
        return vecs.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed(text)[0]


# Convenience alias — swap this to SentenceTransformerEmbeddingService in
# environments where HuggingFace is reachable.
EmbeddingService = HashingEmbeddingService
