from agent import db, memory
from agent.embedding_client import StubEmbeddingClient
from agent.llm_client import StubLLMClient


def _conn():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


class FixedVectorEmbeddingClient:
    def __init__(self, vectors: dict, default: list):
        self._vectors = vectors
        self._default = default

    def embed(self, text: str) -> list:
        return self._vectors.get(text, self._default)


def test_retrieve_returns_needs_input_when_category_empty():
    conn = _conn()
    llm = StubLLMClient()
    llm.register("why do you want", {"category": "motivation"})
    embedder = StubEmbeddingClient()

    result = memory.retrieve(conn, llm, embedder, "why do you want to join us?")

    assert result == {"status": "needs_input", "category": "motivation"}


def test_retrieve_returns_needs_input_with_only_one_candidate():
    conn = _conn()
    llm = StubLLMClient()
    llm.register("why do you want", {"category": "motivation"})
    embedder = StubEmbeddingClient()
    embedder.alias("why do you want to join us?", "why join acme?")
    db.add_memory_entry(
        conn, "why join acme?", "I care about their mission",
        _vec_bytes(embedder.embed("why join acme?")), "motivation",
    )

    result = memory.retrieve(conn, llm, embedder, "why do you want to join us?")

    assert result["status"] == "needs_input"


def test_retrieve_returns_strong_match_when_enough_close_candidates():
    conn = _conn()
    llm = StubLLMClient()
    llm.register("why do you want", {"category": "motivation"})
    embedder = StubEmbeddingClient()
    embedder.alias("why do you want to join us?", "why join?")
    for i in range(2):
        db.add_memory_entry(
            conn, f"past question {i}", f"past answer {i}",
            _vec_bytes(embedder.embed("why join?")), "motivation",
        )

    result = memory.retrieve(conn, llm, embedder, "why do you want to join us?")

    assert result["status"] == "match"
    assert result["confidence"] == "strong"
    assert len(result["candidates"]) == 2


def test_retrieve_returns_weak_match_between_thresholds():
    conn = _conn()
    llm = StubLLMClient()
    llm.register("why do you want", {"category": "motivation"})
    embedder = FixedVectorEmbeddingClient(
        vectors={"why do you want to join us?": [1.0, 0.0]},
        default=[0.6, 0.8],
    )
    for i in range(2):
        db.add_memory_entry(
            conn, f"past question {i}", f"past answer {i}",
            _vec_bytes(embedder.embed(f"past question {i}")), "motivation",
        )

    result = memory.retrieve(conn, llm, embedder, "why do you want to join us?")

    assert result["status"] == "match"
    assert result["confidence"] == "weak"


def test_synthesize_answer_calls_llm_with_candidates():
    llm = StubLLMClient()
    llm.register("why join?", {"answer": "Blended answer"})

    answer = memory.synthesize_answer(llm, "why join?", ["ans A", "ans B"])

    assert answer == "Blended answer"


def test_store_answer_canonicalizes_and_persists():
    conn = _conn()
    llm = StubLLMClient()
    llm.register("raw answer text", {"canonical_answer": "canonical version"})
    embedder = StubEmbeddingClient()

    memory.store_answer(conn, llm, embedder, "why join?", "raw answer text", "motivation")

    rows = db.list_memory_by_category(conn, "motivation")
    assert len(rows) == 1
    assert rows[0]["canonical_answer"] == "canonical version"


def _vec_bytes(vector: list[float]) -> bytes:
    import struct

    return struct.pack(f"{len(vector)}f", *vector)
