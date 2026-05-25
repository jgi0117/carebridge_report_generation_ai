from __future__ import annotations

import os

import gradio as gr

from src.aggregation.monthly import build_monthly_summary
from src.generation.llama_reporter import LlamaReportGenerator
from src.generation.monthly_service import build_monthly_report, list_available_months, load_default_context
from src.ingestion.emotion_loader import load_emotion_probabilities


csv_path, emotion_columns, configs = load_default_context()
llm_config = configs["llm"]["llm"]
generator = LlamaReportGenerator(llm_config)


def load_sample_table():
    """샘플 CSV를 월별 집계표로 변환해 Gradio에 표시합니다."""
    dataframe = load_emotion_probabilities(csv_path)
    return build_monthly_summary(dataframe, emotion_columns)


def generate_monthly_report(month: str):
    """선택한 월의 리포트를 생성하고 집계표/리포트/JSON을 함께 반환합니다."""
    try:
        result = build_monthly_report(
            csv_path=csv_path,
            month=month,
            generator=generator,
            emotion_columns=emotion_columns,
        )
        timed_report = f"{result['report']}\n\n---\n생성 시간: {result['generation_seconds']}초"
        summary_with_time = {
            **result["month_summary"],
            "generation_seconds": result["generation_seconds"],
        }
        return load_sample_table(), timed_report, summary_with_time
    except (RuntimeError, ValueError) as error:
        return load_sample_table(), f"### 리포트 생성 실패\n\n{error}", {"error": str(error)}


with gr.Blocks(title="CareBridge 월간 감정 리포트") as demo:
    gr.Markdown("# CareBridge 월간 감정 리포트")
    gr.Markdown("샘플 감정 확률 CSV를 월별로 집계하고 Llama 3.1 8B로 간단한 케어 리포트를 생성합니다.")

    months = list_available_months(csv_path, emotion_columns)

    with gr.Row():
        month_input = gr.Dropdown(
            choices=months,
            value=months[-1],
            label="리포트 생성 월",
        )
        generate_button = gr.Button("월간 리포트 생성", variant="primary")

    summary_table = gr.Dataframe(
        value=load_sample_table(),
        label="월별 감정 확률 집계",
        interactive=False,
    )
    report_output = gr.Markdown(label="LLM 생성 리포트")
    raw_summary = gr.JSON(label="선택 월 집계 JSON")

    generate_button.click(
        fn=generate_monthly_report,
        inputs=[month_input],
        outputs=[summary_table, report_output, raw_summary],
    )


if __name__ == "__main__":
    demo.launch(
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
