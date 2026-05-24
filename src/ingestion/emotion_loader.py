from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_emotion_probabilities(csv_path: str | Path) -> pd.DataFrame:
    """감정 확률 CSV를 로드하고 날짜 컬럼을 datetime으로 변환합니다."""
    dataframe = pd.read_csv(csv_path)
    if "date" not in dataframe.columns:
        raise ValueError("CSV 파일에는 'date' 컬럼이 필요합니다.")

    dataframe["date"] = pd.to_datetime(dataframe["date"])
    return dataframe.sort_values("date").reset_index(drop=True)


def validate_emotion_columns(dataframe: pd.DataFrame, emotion_columns: list[str]) -> None:
    """리포트 집계에 필요한 감정 컬럼이 모두 있는지 확인합니다."""
    missing_columns = [column for column in emotion_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"CSV에 필요한 감정 컬럼이 없습니다: {missing_columns}")
