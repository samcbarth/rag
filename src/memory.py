import time
from typing import Optional
from .embeddings import EmbeddingService
from .vector_store import VectorStore


class MemoryManager:
    """
    High-level API for storing and retrieving agent memories.

    Memory types
    ------------
    - "semantic"  : facts, knowledge, general information
    - "episodic"  : conversation turns, events, experiences
    - "procedural": how-to instructions, learned behaviors

    Each memory is stored as a vector so that queries retrieve
    the most semantically relevant memories regardless of exact wording.
    """

    def __init__(
        self,
        embed_service: Optional[EmbeddingService] = None,
        persist_dir: Optional[str] = None,
        collection_name: str = "agent_memories",
    ):
        self.embedder = embed_service or EmbeddingService()
        self.store = VectorStore(collection_name=collection_name, persist_dir=persist_dir)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        memory_type: str = "semantic",
        source: str = "user",
        extra_metadata: Optional[dict] = None,
    ) -> str:
        """Embed *content* and persist it. Returns the assigned memory ID."""
        metadata = {
            "type": memory_type,
            "source": source,
            "timestamp": time.time(),
            **(extra_metadata or {}),
        }
        embedding = self.embedder.embed_one(content)
        return self.store.add(text=content, embedding=embedding, metadata=metadata)

    def add_episode(self, role: str, content: str, turn: int = 0) -> str:
        """Convenience wrapper for storing a conversation turn."""
        return self.add(
            content=f"[{role}]: {content}",
            memory_type="episodic",
            source=role,
            extra_metadata={"turn": turn},
        )

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """
        Return the top-n most relevant memories for *query*.
        Optionally filter by memory_type ("semantic", "episodic", "procedural").
        """
        embedding = self.embedder.embed_one(query)
        where = {"type": memory_type} if memory_type else None
        return self.store.search(
            query_embedding=embedding,
            n_results=n_results,
            where=where,
            min_similarity=min_similarity,
        )

    def format_for_prompt(self, memories: list[dict]) -> str:
        """Render retrieved memories as a readable block for injection into prompts."""
        if not memories:
            return ""
        lines = ["[Relevant memories retrieved from vector store]"]
        for i, m in enumerate(memories, 1):
            sim_pct = int(m["similarity"] * 100)
            mtype = m["metadata"].get("type", "?")
            lines.append(f"{i}. ({mtype}, {sim_pct}% match) {m['text']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Manage
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict]:
        return self.store.get_all()

    def clear(self) -> None:
        self.store.clear()

    def count(self) -> int:
        return self.store.count()
