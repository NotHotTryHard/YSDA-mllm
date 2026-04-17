from __future__ import annotations

from typing import Sequence

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


DEFAULT_SENTENCE_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    """Wrapper around sentence-transformer encoder."""

    def __init__(
        self,
        model_name: str | None = DEFAULT_SENTENCE_ENCODER_NAME,
    ) -> None:
        self.model_name = model_name or DEFAULT_SENTENCE_ENCODER_NAME
        self._load_sentence_embedder()

    def _load_sentence_embedder(self):
        self.model = SentenceTransformer(self.model_name)

    def get_embeddings(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
