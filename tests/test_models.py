from agent.models import ApplicationState, FieldSpec


def test_application_state_values_match_db_strings():
    assert ApplicationState.QUEUED.value == "queued"
    assert ApplicationState.AWAITING_HITL.value == "awaiting_hitl"
    assert ApplicationState.DONE.value == "done"


def test_field_spec_defaults():
    spec = FieldSpec(label="Full name", field_type="text")
    assert spec.options == []
    assert spec.required is False
