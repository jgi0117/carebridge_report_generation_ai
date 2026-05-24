from __future__ import annotations

import pandas as pd


def build_monthly_summary(dataframe: pd.DataFrame, emotion_columns: list[str]) -> pd.DataFrame:
    """일별 감정 확률을 월별 평균/주요 감정/변동성 정보로 집계합니다."""
    monthly_source = dataframe.copy()
    monthly_source["month"] = monthly_source["date"].dt.to_period("M").astype(str)

    grouped = monthly_source.groupby("month", as_index=False)
    mean_probabilities = grouped[emotion_columns].mean()
    day_counts = grouped.size().rename(columns={"size": "days"})

    summary = mean_probabilities.merge(day_counts, on="month")
    summary["dominant_emotion"] = summary[emotion_columns].idxmax(axis=1)
    summary["dominant_probability"] = summary[emotion_columns].max(axis=1)
    summary["negative_signal"] = summary[["당황", "분노", "슬픔"]].sum(axis=1)
    summary["positive_signal"] = summary["기쁨"]
    return summary.round(4)


def get_month_summary(monthly_summary: pd.DataFrame, month: str) -> dict:
    """특정 월의 집계 결과를 LLM 프롬프트에 넣기 좋은 dict로 변환합니다."""
    matched = monthly_summary.loc[monthly_summary["month"] == month]
    if matched.empty:
        raise ValueError(f"월별 집계 결과에서 {month} 데이터를 찾을 수 없습니다.")
    return matched.iloc[0].to_dict()


def build_trend_context(monthly_summary: pd.DataFrame, month: str) -> list[dict]:
    """선택 월 이전 흐름을 함께 보여주기 위한 간단한 추세 컨텍스트입니다."""
    months = monthly_summary["month"].tolist()
    month_index = months.index(month)
    start_index = max(0, month_index - 2)
    end_index = min(len(months), month_index + 1)
    return monthly_summary.iloc[start_index:end_index].to_dict(orient="records")
