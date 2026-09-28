import hashlib
import struct
from typing import Protocol


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class StubEmbeddingClient:
    def __init__(self, dims: int = 8) -> None:
        self.dims = dims
        self._aliases: dict[str, str] = {}

    def alias(self, text: str, canonical_text: str) -> None:
        self._aliases[text] = canonical_text

    def embed(self, text: str) -> list[float]:
        key = self._aliases.get(text, text)
        digest = hashlib.sha256(key.encode()).digest()
        raw = struct.unpack(f"{self.dims}i", digest[: self.dims * 4])
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]


class BgeEmbeddingClient:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except ImportError:
            pass

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()
