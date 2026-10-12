from __future__ import annotations

from typing import Any

import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot create embeddings for an empty text list.")
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
