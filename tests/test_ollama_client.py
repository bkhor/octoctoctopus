import pytest
import requests

from agent.llm_client import OllamaClient


def _ollama_reachable() -> bool:
    try:
        requests.get("http://localhost:11434", timeout=1)
        return True
    except requests.exceptions.RequestException:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable at localhost:11434")
def test_ollama_client_returns_valid_json():
    client = OllamaClient()

    result = client.complete_json(
        system_prompt="You classify a single word as either a fruit or not a fruit.",
        user_prompt="apple",
        schema_hint='{"is_fruit": bool}',
    )

    assert isinstance(result, dict)
    assert "is_fruit" in result
    assert result["is_fruit"] is True
