from __future__ import annotations

from time import perf_counter
from pathlib import Path

from src.aggregation.monthly import (
    build_monthly_summary,
    build_trend_context,
    format_summary_for_report,
    format_trend_context_for_report,
    get_month_summary,
)
from src.generation.llama_reporter import LlamaReportGenerator
from src.ingestion.emotion_loader import load_emotion_probabilities, validate_emotion_columns
from src.prompts.monthly_report import build_monthly_report_prompt
from src.utils.config import load_configs, resolve_project_path


def load_default_context() -> tuple[Path, list[str], dict]:
    """기본 샘플 CSV 경로와 감정 컬럼 설정을 읽습니다."""
    configs = load_configs()
    csv_path = resolve_project_path(configs["paths"]["paths"]["sample_emotion_csv"])
    emotion_columns = configs["report"]["report"]["emotion_columns"]
    return csv_path, emotion_columns, configs


def build_monthly_report(
    csv_path: str | Path,
    month: str,
    generator: LlamaReportGenerator,
    emotion_columns: list[str],
) -> dict:
    """CSV 데이터에서 선택 월을 집계하고 Llama 리포트를 생성합니다."""
    dataframe = load_emotion_probabilities(csv_path)
    validate_emotion_columns(dataframe, emotion_columns)

    monthly_summary = build_monthly_summary(dataframe, emotion_columns)
    month_summary = get_month_summary(monthly_summary, month)
    trend_context = build_trend_context(monthly_summary, month)
    probability_keys = emotion_columns + ["dominant_probability", "negative_signal", "positive_signal"]
    prompt_month_summary = format_summary_for_report(month_summary, probability_keys)
    prompt_trend_context = format_trend_context_for_report(trend_context, probability_keys)
    prompt = build_monthly_report_prompt(prompt_month_summary, prompt_trend_context)

    generation_started_at = perf_counter()
    report = generator.generate(prompt)
    generation_seconds = round(perf_counter() - generation_started_at, 2)

    return {
        "month": month,
        "month_summary": month_summary,
        "trend_context": trend_context,
        "report": report,
        "generation_seconds": generation_seconds,
    }


def list_available_months(csv_path: str | Path, emotion_columns: list[str]) -> list[str]:
    """CSV에 포함된 월 목록을 반환합니다."""
    dataframe = load_emotion_probabilities(csv_path)
    validate_emotion_columns(dataframe, emotion_columns)
    monthly_summary = build_monthly_summary(dataframe, emotion_columns)
    return monthly_summary["month"].tolist()
