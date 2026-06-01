"""
RAG Agent demo.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py

Embedding backends
------------------
By default the system uses HashingEmbeddingService (offline, keyword-based).
For semantic/meaning-aware retrieval, switch to SentenceTransformerEmbeddingService
(requires HuggingFace access on first run to download ~90 MB model):

    from src import SentenceTransformerEmbeddingService, MemoryManager
    mem = MemoryManager(embed_service=SentenceTransformerEmbeddingService())
"""

import os
from dotenv import load_dotenv
from src import RAGAgent, MemoryManager, HashingEmbeddingService

load_dotenv()


def demo_vector_memory():
    print("=" * 60)
    print("DEMO 1: Vector memory store (offline, keyword embeddings)")
    print("=" * 60)

    mem = MemoryManager()

    mem.add("User prefers concise bullet-point answers.", memory_type="semantic", source="system")
    mem.add("User is building a SaaS product for restaurant inventory.", memory_type="semantic")
    mem.add("Backend stack: Python, FastAPI, PostgreSQL.", memory_type="semantic")
    mem.add("User dislikes verbose explanations.", memory_type="procedural")
    mem.add_episode("user", "How do I structure the database schema?")
    mem.add_episode("assistant", "Use separate tables: items, suppliers, orders.")

    print(f"\nStored {mem.count()} memories.\n")

    queries = [
        ("Python FastAPI backend", None),
        ("restaurant inventory SaaS", None),
        ("database schema tables", "episodic"),
    ]

    for query, mtype in queries:
        label = f"[type={mtype}] " if mtype else ""
        results = mem.search(query, n_results=3, memory_type=mtype)
        print(f"Query: {label}'{query}'")
        print(mem.format_for_prompt(results) or "  (no results above threshold)")
        print()


def demo_agent():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipping agent demo — set ANTHROPIC_API_KEY to run it.\n")
        return

    print("=" * 60)
    print("DEMO 2: RAG Agent (Claude + vector memory)")
    print("=" * 60)

    agent = RAGAgent(persist_dir="./data/agent_memory", store_episodes=True)

    # Seed long-term memory with facts about the user
    agent.remember("User name is Alex.")
    agent.remember("Alex is building a B2B SaaS for restaurant inventory management.")
    agent.remember("Alex uses Python, FastAPI, and PostgreSQL for the backend.")
    agent.remember("Alex asked about database schema normalisation previously.", memory_type="episodic")

    print(f"\nPre-loaded {agent.memory_count} memories.\n")

    for msg in [
        "What is my name and what product am I building?",
        "Remind me what Python backend stack I use and suggest a schema for inventory items.",
    ]:
        print(f"User: {msg}")
        reply = agent.chat(msg)
        print(f"Agent: {reply}\n")
        print("-" * 40)

    print(f"\nTotal memories after conversation: {agent.memory_count}")


if __name__ == "__main__":
    demo_vector_memory()
    demo_agent()
