import pytest

from agent.llm_client import StubLLMClient


def test_stub_llm_returns_registered_response_on_substring_match():
    stub = StubLLMClient()
    stub.register("classify", {"category": "motivation"})

    result = stub.complete_json(
        system_prompt="sys", user_prompt="please classify: why this company?", schema_hint="{}"
    )

    assert result == {"category": "motivation"}


def test_stub_llm_raises_when_no_response_registered():
    stub = StubLLMClient()

    with pytest.raises(AssertionError):
        stub.complete_json(system_prompt="sys", user_prompt="unregistered", schema_hint="{}")
