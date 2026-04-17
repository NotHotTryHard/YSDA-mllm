from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from embedding import SentenceEmbedder
from preprocessing import Preprocessor
from sentence_transformers import CrossEncoder


class VectorDB:
    """Vector database wrapper."""

    def __init__(
        self,
        path2data: str | Path,
        loaders: Mapping[str, Any],
        chunk_length: int,
        chunk_overlap: int,
        embedder: SentenceEmbedder | None = None,
        db_path: str | Path | None = None,
        collection_name: str = "Embeddings",
    ) -> None:
        self.path2data = Path(path2data)
        self.loaders = dict(loaders)
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        self.db_path = Path(db_path) if db_path else self.path2data.parent / "db"
        self.collection_name = collection_name
        self.sentence_embedder = embedder or SentenceEmbedder()
        self.cross_encoder = None
        self.preprocessor = Preprocessor(
            self.path2data,
            self.loaders,
            self.chunk_length,
            self.chunk_overlap,
        )
        self._start_db()

    def _start_db(self) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def _reset_collection(self) -> None:
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def preprocess_data(self) -> list[Document]:
        return self.preprocessor.preprocess_data()

    def get_embeddings(self, chunks: list[Document]) -> np.ndarray:
        return self.sentence_embedder.get_embeddings(chunks)

    def get_uuids(self, chunks):
        uuids = []
        for chunk in chunks:
            metadata = chunk.metadata or {}
            key = "|".join(
                [
                    str(metadata.get("source", "")),
                    str(metadata.get("page", "")),
                    str(metadata.get("start_index", "")),
                    chunk.page_content,
                ]
            )
            uuids.append(sha256(key.encode("utf-8")).hexdigest())
        return uuids

    def fill_db(self, *, reset_collection=True):
        if reset_collection:
            self._reset_collection()

        chunks = self.preprocess_data()
        embeddings = self.get_embeddings(chunks)
        uuids = self.get_uuids(chunks)

        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        existing_ids = set(self.collection.get()["ids"])
        unique_ids = []
        unique_embeddings = []
        unique_docs = []
        unique_meta = []

        seen_ids = set()
        for uid, emb, doc, meta in zip(uuids, embeddings, documents, metadatas):
            if uid in existing_ids or uid in seen_ids:
                continue
            seen_ids.add(uid)
            unique_ids.append(uid)
            unique_embeddings.append(emb.tolist())
            unique_docs.append(doc)
            unique_meta.append(meta)

        if unique_ids:
            self.collection.add(
                ids=unique_ids,
                embeddings=unique_embeddings,
                documents=unique_docs,
                metadatas=unique_meta,
            )
        return len(unique_ids)

    def remove_collection(self) -> None:
        self._reset_collection()

    def _init_cross_encoder(self, cross_encoder_name: str) -> None:
        self.cross_encoder = CrossEncoder(cross_encoder_name)

    def rerank(self, prompt, documents, top_k, cross_encoder_name):
        if self.cross_encoder is None:
            self._init_cross_encoder(cross_encoder_name)
        pairs = [[prompt, doc] for doc in documents]
        scores = np.asarray(self.cross_encoder.predict(pairs), dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return np.asarray(documents)[order], scores[order]

    def query(
        self,
        prompt,
        threshold=0.3,
        top_n=5,
        top_k=3,
        use_rerank=True,
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        stored = self.collection.get(include=["embeddings", "documents"])
        stored_ids = stored.get("ids") or []
        if not stored_ids:
            return None, None

        stored_docs = stored.get("documents") or []
        stored_embeddings = np.asarray(stored.get("embeddings"), dtype=np.float32)

        prompt_embedding = self.sentence_embedder.get_embeddings([Document(page_content=prompt)])
        similarities = cosine_similarity(prompt_embedding, stored_embeddings)[0]

        mask = similarities >= threshold
        if not mask.any():
            return None, None

        filtered_scores = similarities[mask]
        filtered_docs = np.asarray(stored_docs)[mask]

        order = np.argsort(-filtered_scores)[:top_n]
        documents = filtered_docs[order]
        scores = filtered_scores[order]

        if use_rerank:
            documents, scores = self.rerank(prompt, documents.tolist(), top_k, cross_encoder_name)
        return documents, scores

