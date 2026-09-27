import pytest

from agent.config import DEFAULTS, load_config


def test_load_config_returns_defaults_when_file_missing(tmp_path):
    config = load_config(str(tmp_path / "nonexistent.yaml"))

    assert config == DEFAULTS


def test_load_config_overrides_one_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("ollama_model: mistral:7b\n")

    config = load_config(str(path))

    assert config["ollama_model"] == "mistral:7b"
    assert config["db_path"] == DEFAULTS["db_path"]


def test_load_config_overrides_all_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "db_path: /tmp/custom.db\n"
        "profile_path: /tmp/profile.yaml\n"
        "ollama_base_url: http://example.com:1234\n"
        "ollama_model: custom-model\n"
        "bge_model: custom-bge\n"
    )

    config = load_config(str(path))

    assert config == {
        "db_path": "/tmp/custom.db",
        "profile_path": "/tmp/profile.yaml",
        "ollama_base_url": "http://example.com:1234",
        "ollama_model": "custom-model",
        "bge_model": "custom-bge",
    }


def test_load_config_ignores_unknown_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("unknown_key: surprise\ndb_path: /tmp/custom.db\n")

    config = load_config(str(path))

    assert "unknown_key" not in config
    assert config["db_path"] == "/tmp/custom.db"


def test_load_config_treats_empty_file_as_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")

    config = load_config(str(path))

    assert config == DEFAULTS


def test_load_config_raises_on_non_mapping_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError):
        load_config(str(path))
