from typing import Protocol


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
