from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation.llama_reporter import LlamaReportGenerator
from src.generation.monthly_service import build_monthly_report, list_available_months, load_default_context


csv_path, emotion_columns, configs = load_default_context()
llm_config = configs["llm"]["llm"]
generator = LlamaReportGenerator(llm_config)

app = FastAPI(
    title="CareBridge Report Generation API",
    description="감정 확률 데이터를 기반으로 월간 케어 리포트를 생성하는 API입니다.",
    version="0.1.0",
)


class MonthlyReportRequest(BaseModel):
    month: str = Field(..., examples=["2026-05"])
    csv_path: str | None = Field(default=None, description="생략하면 샘플 감정 확률 CSV를 사용합니다.")


@app.get("/")
def root() -> dict:
    return {"service": "carebridge_report_generation_ai", "status": "running"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_id": llm_config["model_id"],
        "sample_csv": str(csv_path),
        "model_cache": generator.cache_status(),
        "runtime": generator.runtime_status(),
    }


@app.get("/months")
def months() -> dict:
    return {"months": list_available_months(csv_path, emotion_columns)}


@app.post("/reports/monthly")
def create_monthly_report(request: MonthlyReportRequest) -> dict:
    target_csv_path = Path(request.csv_path) if request.csv_path else csv_path
    try:
        return build_monthly_report(
            csv_path=target_csv_path,
            month=request.month,
            generator=generator,
            emotion_columns=emotion_columns,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
