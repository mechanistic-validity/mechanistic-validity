"""pydantic-settings configuration — unified CLI / env / YAML config.

Precedence: CLI flags > env vars (MV_*) > mv_config.yaml > defaults.

Requires the 'cli' optional dependency group:
    pip install mechanistic-validity[cli]
"""
from __future__ import annotations

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class MechValSettings(BaseSettings):
        output_dir: str = "./results"
        device: str = "cpu"
        model_name: str = "gpt2"
        wandb_project: str = ""
        n_prompts: int = 40
        seed: int = 42

        model_config = SettingsConfigDict(
            env_prefix="MV_",
            yaml_file="mv_config.yaml",
            yaml_file_encoding="utf-8",
        )

except ImportError:
    from pydantic import BaseModel

    class MechValSettings(BaseModel):  # type: ignore[no-redef]
        output_dir: str = "./results"
        device: str = "cpu"
        model_name: str = "gpt2"
        wandb_project: str = ""
        n_prompts: int = 40
        seed: int = 42
