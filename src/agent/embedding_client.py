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
