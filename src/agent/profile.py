import yaml


def load_profile(path: str) -> dict[str, str]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping of fields")
    return data
