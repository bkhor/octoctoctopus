import json
from typing import Protocol

import requests


class LLMClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str, schema_hint: str) -> dict:
        ...


class StubLLMClient:
    def __init__(self) -> None:
        self._responses: list[tuple[str, dict]] = []

    def register(self, prompt_contains: str, response: dict) -> None:
        self._responses.append((prompt_contains, response))

    def complete_json(self, system_prompt: str, user_prompt: str, schema_hint: str) -> dict:
        for needle, response in self._responses:
            if needle in user_prompt:
                return response
        raise AssertionError(f"StubLLMClient: no registered response matches prompt: {user_prompt!r}")


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b") -> None:
        self.base_url = base_url
        self.model = model

    def complete_json(self, system_prompt: str, user_prompt: str, schema_hint: str) -> dict:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"{system_prompt}\n\nRespond with JSON matching this schema: {schema_hint}",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                "format": "json",
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return json.loads(content)
