# -*- coding: utf-8 -*-
"""Configuration loader for the LangChain integration.

Loads settings from ``config/config.yaml`` and environment variables.
Uses ``python-dotenv`` to read a ``.env`` file located at the project root.
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[2]
DOTENV_PATH = ROOT_DIR / ".env"
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH)

# Load YAML config
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config_yaml = yaml.safe_load(f)

class Settings:
    """Simple settings container.

    Attributes are populated from the YAML file with fall‑backs to environment variables.
    """

    # Model settings
    model_name: str = _config_yaml.get("model", {}).get("name", "groq/gemma-2b-it")
    groq_api_key: str = os.getenv("GROQ_API_KEY", _config_yaml.get("groq_api_key", ""))

    # Retrieval settings
    weaviate_url: str = _config_yaml.get("retrieval", {}).get("weaviate_url", "http://localhost:8080")
    use_hybrid: bool = _config_yaml.get("retrieval", {}).get("use_hybrid", True)

    # Memory settings
    short_term_window: int = _config_yaml.get("memory", {}).get("short_term_window", 5)
    decay_rate: float = _config_yaml.get("memory", {}).get("decay_rate", 0.1)

    # Cache settings
    cache_enabled: bool = _config_yaml.get("cache", {}).get("enabled", True)
    cache_backend: str = _config_yaml.get("cache", {}).get("backend", "memory")

    # Logging
    logging_level: str = _config_yaml.get("logging", {}).get("level", "INFO")

settings = Settings()
