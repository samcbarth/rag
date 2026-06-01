import uuid
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    ChromaDB-backed vector store.

    Each record holds:
      - id        : unique string key
      - embedding : dense float vector
      - document  : raw text content
      - metadata  : arbitrary key/value pairs (must be str/int/float/bool)
    """

    def __init__(self, collection_name: str = "memories", persist_dir: Optional[str] = None):
        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        doc_id = doc_id or str(uuid.uuid4())
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )
        return doc_id

    def update(
        self,
        doc_id: str,
        text: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        kwargs: dict = {"ids": [doc_id]}
        if text is not None:
            kwargs["documents"] = [text]
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        if metadata is not None:
            kwargs["metadatas"] = [metadata]
        self._collection.update(**kwargs)

    def delete(self, doc_id: str) -> None:
        self._collection.delete(ids=[doc_id])

    def clear(self) -> None:
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """
        Return up to n_results records ordered by cosine similarity (desc).
        Each result dict: {id, text, metadata, similarity}.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(self._collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results = []
        for doc_id, text, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            similarity = 1.0 - dist  # cosine distance → similarity
            if similarity >= min_similarity:
                results.append({"id": doc_id, "text": text, "metadata": meta, "similarity": similarity})

        return results

    def get_all(self) -> list[dict]:
        raw = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": doc_id, "text": text, "metadata": meta}
            for doc_id, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]

    def count(self) -> int:
        return self._collection.count()
