from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

from src.generation.llama_reporter import LlamaReportGenerator
from src.utils.config import load_configs


ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hugging Face 모델을 로컬 캐시에 다운로드합니다.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="네트워크 다운로드 없이 현재 로컬 캐시 상태만 확인합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_configs()
    llm_config = configs["llm"]["llm"]
    model_id = llm_config["model_id"]
    token_env = llm_config.get("hf_token_env", "HF_TOKEN")
    token = os.getenv(token_env) or None

    if not token and not args.local_files_only:
        raise RuntimeError(f"{token_env} 환경변수가 비어 있습니다. .env에 Hugging Face read token을 설정해주세요.")

    snapshot_download(
        repo_id=model_id,
        token=token,
        allow_patterns=ALLOW_PATTERNS,
        local_files_only=args.local_files_only,
    )

    generator = LlamaReportGenerator(llm_config)
    status = generator.cache_status()
    print(f"model_id={status['model_id']}")
    print(f"cache_dir={status['cache_dir']}")
    print(f"is_complete={status['is_complete']}")
    if status["missing_files"]:
        print(f"missing_files={status['missing_files']}")


if __name__ == "__main__":
    main()
