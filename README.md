# CareBridge Report Generation AI

CareBridge Report Generation AI는 CareBridge 서비스에서 축적되는 감정 인식 결과, 행동 분석 결과, 케어 일지 등의 데이터를 요약하고 LLM을 통해 돌봄 리포트를 생성하기 위한 실험용 프로젝트입니다.

초기 MVP는 `emotion_recognition`의 이미지별 감정 확률 데이터를 기간별로 집계하고, 요약된 JSON을 기반으로 리포트를 생성하는 방향으로 시작합니다. 이후 케어 일지, 상담 기록, 행동 분석 로그 등 다른 데이터를 결합할 수 있도록 구조를 분리해 둡니다.

## 프로젝트 구조

```text
src/
  ingestion/       외부 데이터 입력 및 로딩
  aggregation/     기간별/사용자별 지표 집계
  prompts/         LLM 프롬프트 템플릿
  generation/      리포트 생성 로직
  api/             FastAPI 등 서비스 엔드포인트
  utils/           공통 유틸리티

configs/           경로, 리포트, LLM 설정
data/              로컬 입력 데이터
notebooks/         실험 노트북
output/            생성 리포트 및 중간 산출물
requirements.txt
README.md
```

## 환경 설정

```powershell
cd C:\Users\codeit44\Desktop\side_project\report_generation
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 방향

1. 감정 인식 결과 CSV/DB를 입력으로 받습니다.
2. 일/주/월 단위로 감정 분포와 변화 지표를 집계합니다.
3. 리포트 생성용 summary JSON을 만듭니다.
4. LLM에 summary JSON을 전달해 리포트를 생성합니다.
5. 생성된 리포트와 사람이 수정한 최종 리포트를 축적해 추후 평가셋 또는 fine-tuning 데이터로 활용합니다.
