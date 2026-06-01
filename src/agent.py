import os
from typing import Optional
import anthropic
from .memory import MemoryManager


class RAGAgent:
    """
    A Claude-powered agent with persistent vector memory.

    On each turn the agent:
      1. Embeds the user message and retrieves the most relevant memories.
      2. Injects those memories into the system prompt as grounding context.
      3. Calls Claude to generate a response.
      4. Optionally stores the user turn and assistant response as episodic memories.
    """

    DEFAULT_SYSTEM = (
        "You are a helpful AI assistant with access to a long-term memory store. "
        "When relevant memories are provided, use them to give accurate, personalised answers. "
        "If no memories are relevant, answer from your general knowledge."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        system_prompt: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None,
        persist_dir: Optional[str] = None,
        n_memories: int = 5,
        min_similarity: float = 0.3,
        store_episodes: bool = True,
    ):
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.memory = memory_manager or MemoryManager(persist_dir=persist_dir)
        self.n_memories = n_memories
        self.min_similarity = min_similarity
        self.store_episodes = store_episodes
        self._history: list[dict] = []  # in-context conversation turns
        self._turn = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send a message, retrieve memory, generate a response."""
        memories = self.memory.search(
            query=user_message,
            n_results=self.n_memories,
            min_similarity=self.min_similarity,
        )
        memory_block = self.memory.format_for_prompt(memories)

        system = self.system_prompt
        if memory_block:
            system = f"{system}\n\n{memory_block}"

        self._history.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=self._history,
        )
        assistant_text = response.content[0].text
        self._history.append({"role": "assistant", "content": assistant_text})

        if self.store_episodes:
            self.memory.add_episode("user", user_message, turn=self._turn)
            self.memory.add_episode("assistant", assistant_text, turn=self._turn)

        self._turn += 1
        return assistant_text

    def remember(self, fact: str, memory_type: str = "semantic", source: str = "explicit") -> str:
        """Explicitly store a fact or instruction in long-term memory."""
        mem_id = self.memory.add(content=fact, memory_type=memory_type, source=source)
        return mem_id

    def recall(self, query: str, n: int = 5) -> list[dict]:
        """Search memory and return raw result dicts."""
        return self.memory.search(query, n_results=n, min_similarity=self.min_similarity)

    def reset_conversation(self) -> None:
        """Clear in-context history (long-term memory is unaffected)."""
        self._history.clear()
        self._turn = 0

    def clear_memory(self) -> None:
        """Wipe all stored memories."""
        self.memory.clear()

    @property
    def memory_count(self) -> int:
        return self.memory.count()
