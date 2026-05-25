# CareBridge Report Generation AI

CareBridge Report Generation AI는 감정 인식 결과, 행동 분석 결과, 케어 일지 같은 데이터를 바탕으로 보호자와 케어 매니저가 읽기 쉬운 리포트를 자동 생성하기 위한 실험 프로젝트입니다.

현재 MVP는 `emotion_recognition`에서 생성되는 이미지 기반 감정 확률 데이터를 월별로 집계하고, `Llama 3.1 8B Instruct`를 이용해 간단한 월간 감정 리포트를 생성합니다. 이후에는 케어 일지, 상담 기록, 행동 분석 로그를 함께 넣어 더 풍부한 케어 리포트로 확장할 수 있도록 구조를 분리했습니다.

## 프로젝트 구조

```text
src/
  ingestion/       CSV/DB 등 입력 데이터 로딩
  aggregation/     일별 데이터를 월별 리포트 지표로 집계
  prompts/         LLM 프롬프트 템플릿
  generation/      Llama 기반 리포트 생성 로직
  api/             FastAPI, Gradio 실행 코드
  utils/           설정 파일 로딩 등 공통 유틸리티

configs/           경로, 리포트, LLM 설정
data/              로컬 입력 데이터
notebooks/         실험 노트북
output/            생성 리포트와 중간 산출물
```

## 환경 설정

로컬 Windows 환경에서는 기존 `requirements.txt`를 사용합니다.

```powershell
cd C:\Users\codeit44\Desktop\side_project\report_generation
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Llama 3.1은 Hugging Face에서 접근 권한 동의가 필요한 gated 모델입니다. 모델 페이지에서 라이선스와 접근 요청을 승인한 뒤, 필요하면 `HF_TOKEN` 환경 변수를 설정하세요.

```powershell
$env:HF_TOKEN="your_huggingface_token"
```

또는 프로젝트 루트의 `.env` 파일에 아래처럼 저장할 수 있습니다. `.env`는 Git에서 제외되어 push되지 않습니다.

```text
HF_TOKEN=your_huggingface_token
```

fine-grained token을 사용하는 경우 Hugging Face token settings에서 아래 권한을 켜야 gated 모델 파일을 받을 수 있습니다.

```text
Read access to contents of all public gated repos you can access
```

LLM 설정 파일은 기본적으로 `configs/llm.yaml`을 사용합니다. 다른 환경용 설정을 쓰고 싶을 때는 `REPORT_LLM_CONFIG` 환경변수만 바꾸면 됩니다.

```powershell
$env:REPORT_LLM_CONFIG="configs/llm.yaml"
```

## 모델 다운로드

처음 실행하는 환경에서는 모델을 먼저 로컬 Hugging Face 캐시에 받아두는 것을 권장합니다.

```powershell
python -m src.generation.download_model
```

다운로드가 끝났는지 확인하려면 네트워크 없이 캐시 상태만 검사할 수 있습니다.

```powershell
python -m src.generation.download_model --local-files-only
```

정상이라면 `is_complete=True`가 출력됩니다.

## Colab GPU 실행

Colab에서는 기존 CUDA용 `torch`를 유지하는 것이 안전하므로 `requirements-colab.txt`를 사용합니다. 이 파일에는 `torch`를 넣지 않았습니다.

```bash
pip install -r requirements-colab.txt
```

Colab 노트북에서 토큰과 Colab용 LLM 설정을 지정합니다.

```python
import os

os.environ["HF_TOKEN"] = "your_huggingface_token"
os.environ["REPORT_LLM_CONFIG"] = "configs/llm_colab.yaml"
os.environ["GRADIO_SHARE"] = "true"
```

GPU가 잡혔는지 먼저 확인합니다.

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

그 다음 모델을 다운로드하고 캐시 상태를 확인합니다.

```bash
python -m src.generation.download_model
python -m src.generation.download_model --local-files-only
```

`configs/llm_colab.yaml`은 `local_files_only: false`, `torch_dtype: float16`으로 설정되어 있어 Colab GPU에서 모델을 다운로드하고 GPU 메모리에 올리는 흐름에 맞춰져 있습니다.

Colab에서 Gradio 화면을 보려면 public link가 필요합니다. 위처럼 `GRADIO_SHARE=true`를 설정한 뒤 실행하면 `https://...gradio.live` 형태의 주소가 출력됩니다.

```bash
python -m src.api.app_gradio
```

## Gradio 실행

```powershell
python -m src.api.app_gradio
```

브라우저에서 샘플 CSV의 월별 집계표를 확인하고, 선택한 월에 대한 LLM 리포트를 생성할 수 있습니다.

로컬 기본 설정은 `configs/llm.yaml`의 `local_files_only: true`로 되어 있습니다. 모델 다운로드가 완료된 뒤에는 Hugging Face 서버 확인 없이 로컬 캐시만 사용하므로, 네트워크가 막힌 환경에서도 실행할 수 있습니다. Colab처럼 처음 모델을 받는 환경에서는 `REPORT_LLM_CONFIG=configs/llm_colab.yaml`을 지정하세요.

로컬 PC에서 실행할 때는 `http://127.0.0.1:7860`으로 접속하면 됩니다. Colab에서 실행할 때는 이 주소가 Colab 내부 주소라 직접 접속할 수 없고, `GRADIO_SHARE=true`로 생성된 public link를 사용해야 합니다.

현재 기본 `max_new_tokens`는 512입니다. CPU 환경에서는 Llama 3.1 8B 추론이 매우 느릴 수 있으므로, 빠른 MVP 확인이 목적이라면 이 값을 낮추거나 3B급 모델로 바꿔 먼저 기능 흐름을 검증하는 것이 좋습니다.

## FastAPI 실행

```powershell
uvicorn src.api.app_api:app --reload
```

주요 엔드포인트:

- `GET /health`: 서비스와 모델 설정 확인
- `GET /months`: 샘플 CSV에 포함된 월 목록 조회
- `POST /reports/monthly`: 특정 월 리포트 생성

`GET /health`는 모델 캐시가 완성되었는지, CUDA 사용 가능 여부, 현재 생성 토큰 수 설정을 함께 반환합니다. 백엔드 연동 전 이 값을 먼저 확인하면 모델 다운로드/환경 문제를 빠르게 분리할 수 있습니다.

요청 예시:

```json
{
  "month": "2026-05"
}
```
