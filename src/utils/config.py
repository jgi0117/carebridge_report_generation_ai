from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# 프로젝트 어디에서 실행해도 configs/data 경로를 동일하게 찾기 위한 기준 경로입니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(env_path: str | Path = ".env") -> None:
    """로컬 .env 파일의 KEY=VALUE 값을 환경변수로 올립니다."""
    dotenv_path = resolve_project_path(env_path)
    if not dotenv_path.exists():
        return

    with dotenv_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_project_path(path: str | Path) -> Path:
    """상대 경로는 프로젝트 루트 기준으로, 절대 경로는 그대로 반환합니다."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일을 읽어 dict로 반환합니다."""
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_configs() -> dict[str, Any]:
    """리포트 생성에 필요한 설정 파일들을 한 번에 읽습니다."""
    load_dotenv()
    llm_config_path = os.getenv("REPORT_LLM_CONFIG", "configs/llm.yaml")
    return {
        "paths": load_yaml("configs/paths.yaml"),
        "report": load_yaml("configs/report.yaml"),
        "llm": load_yaml(llm_config_path),
    }
