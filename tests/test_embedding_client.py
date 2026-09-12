from agent.embedding_client import StubEmbeddingClient


def test_stub_embedding_is_deterministic():
    stub = StubEmbeddingClient()
    assert stub.embed("hello") == stub.embed("hello")


def test_stub_embedding_alias_makes_vectors_identical():
    stub = StubEmbeddingClient()
    stub.alias("Tell me about a challenge", "Describe a difficult project")

    assert stub.embed("Tell me about a challenge") == stub.embed("Describe a difficult project")


def test_stub_embedding_different_text_differs():
    stub = StubEmbeddingClient()
    assert stub.embed("apples") != stub.embed("oranges")
