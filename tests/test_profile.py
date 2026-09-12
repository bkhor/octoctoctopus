from pathlib import Path

from agent.profile import load_profile

FIXTURE = Path(__file__).parent / "fixtures" / "sample_profile.yaml"


def test_load_profile_returns_flat_dict():
    data = load_profile(str(FIXTURE))
    assert data["full_name"] == "Jane Doe"
    assert data["work_authorization"] == "Authorized to work in the US"


def test_load_profile_missing_file_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_profile("/nonexistent/profile.yaml")
