from pathlib import Path

import yaml

DEFAULTS = {
    "db_path": "./octo.db",
    "profile_path": "./profile.yaml",
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "qwen2.5:7b",
    "bge_model": "BAAI/bge-small-en-v1.5",
}


def load_config(path: str) -> dict:
    config = dict(DEFAULTS)
    file_path = Path(path)
    if not file_path.exists():
        return config
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return config
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping of config keys")
    for key in DEFAULTS:
        if key in data:
            config[key] = data[key]
    return config
