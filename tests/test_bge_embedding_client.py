import pytest

pytest.importorskip("sentence_transformers")

from agent.embedding_client import BgeEmbeddingClient


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)


def test_bge_embedding_scores_similar_questions_higher_than_unrelated():
    client = BgeEmbeddingClient()

    a = client.embed("What is your greatest strength?")
    b = client.embed("What would you say is your biggest strength?")
    c = client.embed("What is the capital of France?")

    assert _cosine(a, b) > _cosine(a, c)
