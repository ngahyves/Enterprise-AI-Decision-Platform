# src/common/config_manager.py
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Loading env variables
load_dotenv()


class Settings(BaseSettings):
    """Manages environment variables"""

    groq_api_key: str
    google_api_key: str
    llama_cloud_api_key: str

    qdrant_host: str
    qdrant_port: int
    qdrant_collection_name: str

    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: int

    gcp_project_id: str
    gcp_region: str
    gcs_bucket_dvc: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# 2. Loading config variables
def load_yaml_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the parameters in config.yaml file"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found : {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Loading instances
env_settings = Settings()
config = load_yaml_config()
