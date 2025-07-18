import os
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config():
    env = os.getenv("APP_ENV", "dev")  # default to dev
    config_path = BASE_DIR / "config"

    def read_yaml(file):
        with open(config_path / file, "r") as f:
            return yaml.safe_load(f)

    base = read_yaml("base.yaml")
    env_config = read_yaml(f"{env}.yaml")

    # merge with environment-specific overriding base
    config = {**base, **env_config}

    # Expand relative paths
    for key, val in config.items():
        if "PATH" in key or "DIR" in key:
            config[key] = str((BASE_DIR / val).resolve())

    return config
