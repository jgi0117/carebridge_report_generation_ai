from __future__ import annotations

import argparse
from pathlib import Path

from src.generation.llama_reporter import LlamaReportGenerator
from src.generation.monthly_service import build_monthly_report, list_available_months, load_default_context
from src.utils.config import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="월간 감정 리포트를 CLI에서 생성합니다.")
    parser.add_argument(
        "--month",
        help="리포트를 생성할 월입니다. 예: 2026-05. 생략하면 CSV의 마지막 월을 사용합니다.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="입력 감정 확률 CSV 경로입니다. 생략하면 configs/paths.yaml의 샘플 CSV를 사용합니다.",
    )
    parser.add_argument(
        "--output",
        help="생성된 리포트를 저장할 Markdown 파일 경로입니다. 생략하면 output/reports에 저장합니다.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="파일 저장 없이 콘솔에만 출력합니다.",
    )
    return parser.parse_args()


def resolve_output_path(month: str, configs: dict, output: str | None) -> Path:
    """CLI 출력 경로를 결정하고 상위 폴더를 준비합니다."""
    if output:
        output_path = resolve_project_path(output)
    else:
        report_output_dir = resolve_project_path(configs["paths"]["paths"]["report_output"])
        output_path = report_output_dir / f"{month}_monthly_report.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def build_markdown(result: dict) -> str:
    """리포트 본문과 생성 시간을 저장용 Markdown으로 묶습니다."""
    return f"{result['report']}\n\n---\n생성 시간: {result['generation_seconds']}초\n"


def main() -> None:
    args = parse_args()
    default_csv_path, emotion_columns, configs = load_default_context()
    csv_path = resolve_project_path(args.csv_path) if args.csv_path else default_csv_path
    month = args.month or list_available_months(csv_path, emotion_columns)[-1]

    generator = LlamaReportGenerator(configs["llm"]["llm"])
    result = build_monthly_report(
        csv_path=csv_path,
        month=month,
        generator=generator,
        emotion_columns=emotion_columns,
    )
    markdown = build_markdown(result)

    print(markdown)
    if not args.no_save:
        output_path = resolve_output_path(month, configs, args.output)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"저장 경로: {output_path}")


if __name__ == "__main__":
    main()
