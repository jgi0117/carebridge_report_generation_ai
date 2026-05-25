from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts.monthly_report import SYSTEM_PROMPT


class LlamaReportGenerator:
    """Llama 3.1 Instruct 모델을 로드하고 월간 리포트 문장을 생성합니다."""

    def __init__(self, llm_config: dict[str, Any]) -> None:
        self.model_id = llm_config["model_id"]
        self.temperature = float(llm_config.get("temperature", 0.2))
        self.max_new_tokens = int(llm_config.get("max_new_tokens", 700))
        self.device_map = llm_config.get("device_map", "auto")
        self.torch_dtype = self._resolve_runtime_dtype(llm_config)
        self.low_cpu_mem_usage = bool(llm_config.get("low_cpu_mem_usage", True))
        self.trust_remote_code = bool(llm_config.get("trust_remote_code", False))
        self.local_files_only = bool(llm_config.get("local_files_only", True))
        self.hf_token_env = llm_config.get("hf_token_env", "HF_TOKEN")
        self.hf_token = os.getenv(self.hf_token_env) or None
        self._tokenizer = None
        self._model = None

    def _resolve_dtype(self, dtype_name: str):
        """YAML의 dtype 문자열을 torch dtype으로 변환합니다."""
        if dtype_name == "auto":
            return "auto"
        return getattr(torch, dtype_name)

    def _resolve_runtime_dtype(self, llm_config: dict[str, Any]):
        """GPU/CPU 환경에 맞는 dtype을 선택합니다."""
        if torch.cuda.is_available():
            return self._resolve_dtype(llm_config.get("torch_dtype", "auto"))
        return self._resolve_dtype(llm_config.get("cpu_torch_dtype", llm_config.get("torch_dtype", "auto")))

    def _build_model_load_kwargs(self) -> dict[str, Any]:
        """CPU 환경에서 accelerate disk offload가 걸리지 않도록 로딩 옵션을 구성합니다."""
        kwargs = {
            "token": self.hf_token,
            "local_files_only": self.local_files_only,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": self.trust_remote_code,
        }
        if torch.cuda.is_available() and self.device_map:
            kwargs["device_map"] = self.device_map
        return kwargs

    def _build_hf_access_error(self) -> RuntimeError:
        """gated 모델 접근 권한 문제를 백엔드/화면에서 바로 이해할 수 있게 바꿉니다."""
        return RuntimeError(
            "Llama 3.1 8B 모델 접근에 실패했습니다. Hugging Face fine-grained token을 사용한다면 "
            "token settings에서 'Read access to contents of all public gated repos you can access' "
            "권한을 켜거나, gated repository 접근이 가능한 read token을 새로 발급해 .env의 "
            f"{self.hf_token_env}에 넣어주세요."
        )

    def _build_cache_error(self) -> RuntimeError:
        """로컬 캐시 모드에서 필요한 파일이 없을 때 보여줄 안내입니다."""
        return RuntimeError(
            "로컬 캐시에서 Llama 3.1 8B 파일을 찾지 못했습니다. 인터넷 연결이 가능한 환경에서 "
            "모델 다운로드를 먼저 완료하거나, configs/llm.yaml의 local_files_only를 false로 바꿔 "
            "Hugging Face에서 다시 받을 수 있게 해주세요."
        )

    def cache_status(self) -> dict[str, Any]:
        """모델 파일이 로컬 Hugging Face 캐시에 어느 정도 준비됐는지 확인합니다."""
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_dir = cache_root / f"models--{self.model_id.replace('/', '--')}"
        expected_weight_files = [f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)]
        expected_support_files = [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        existing_files = []
        if model_cache_dir.exists():
            existing_files = [path.name for path in model_cache_dir.rglob("*") if path.is_file()]

        expected_files = expected_weight_files + expected_support_files

        return {
            "model_id": self.model_id,
            "local_files_only": self.local_files_only,
            "cache_dir": str(model_cache_dir),
            "expected_files": expected_files,
            "existing_files": sorted(set(existing_files)),
            "missing_files": [file_name for file_name in expected_files if file_name not in existing_files],
            "is_complete": all(file_name in existing_files for file_name in expected_files),
        }

    def runtime_status(self) -> dict[str, Any]:
        """서버 환경과 모델 로드 상태를 API health에서 확인할 수 있게 반환합니다."""
        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "loaded": self._model is not None,
            "max_new_tokens": self.max_new_tokens,
            "torch_dtype": str(self.torch_dtype),
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "device_map": getattr(self._model, "hf_device_map", None) if self._model is not None else None,
        }

    def _load_model(self) -> None:
        """첫 생성 요청 때 모델을 한 번만 로드하고 이후 요청에서는 재사용합니다."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **self._build_model_load_kwargs(),
            )
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as error:
            message = str(error)
            if "public gated repositories" in message or "403 Forbidden" in message:
                raise self._build_hf_access_error() from error
            if "Cannot find" in message or "couldn't connect" in message or "local files" in message:
                raise self._build_cache_error() from error
            if "paging file is too small" in message or "페이징 파일이 너무 작습니다" in message or "os error 1455" in message:
                raise RuntimeError(
                    "모델 로드 중 Windows 페이지 파일/메모리가 부족합니다. "
                    "로컬 CPU에서는 Llama 3.1 8B가 매우 무거우므로 configs/llm.yaml의 torch_dtype을 auto로 유지하고, "
                    "가능하면 GPU 환경이나 더 작은 모델을 사용하세요."
                ) from error
            raise

    def generate(self, user_prompt: str) -> str:
        """시스템 프롬프트와 사용자 프롬프트를 chat template으로 변환해 리포트를 생성합니다."""
        self._load_model()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
