# baseline_eval — Ours(Depth+PPS) vs VLM-only 비교 평가

본문(`llm/`, `vision/`)을 **전혀 수정하지 않는** 독립 평가 폴더입니다.
루트 `.env`의 `OPENAI_API_KEY`만 읽어 씁니다.

## 핵심 주장 (이 실험이 증명하려는 것)

> 객체 검출(YOLO)은 동일하게 두고, **거리 추정과 PPS 실행가능성 판단**을
> `기하(Depth Anything V2 + 3D)`로 하느냐 `VLM 직접(GPT-4V)`으로 하느냐만 바꾼다.
> → 기하 방식이 더 **정확·안정·저렴**하다.

- 통제변인: 프레임, YOLO 박스, VLM 종류
- 조작변인: **거리/결정의 출처** (기하 vs VLM)

## 폴더 구성

| 파일 | 역할 | 지금 실행? |
|---|---|---|
| `metrics.py` | MAE/RMSE/AbsRel/F1/FalseTrig/flip-rate + McNemar·Wilcoxon·부트스트랩·Bland-Altman (numpy 전용) | ✅ 스모크테스트 내장 |
| `run_ours_from_log.py` | 기존 `data/dist_log*.csv`로 Ours 거리 정밀도 산출 | ✅ 이미지 불필요 |
| `collect_frames.py` | 웹캠+YOLO로 공정비교용 프레임·GT 수집 → `manifest.csv` | ⏳ 웹캠 필요 |
| `vlm_baseline.py` | 같은 프레임을 GPT-4V에 직접 질의(거리·결정, N회 반복) | ⏳ 이미지 필요 |
| `run_ours.py` | **본문을 import만** 해서 서버와 동일 경로로 프레임별 person↔target 지면거리 산출 | ⏳ 이미지 필요 |
| `compare.py` | Ours vs VLM 비교표(Main Table) + Bland-Altman/층화 그림 | ⏳ 결과 CSV 필요 |
| `config.py` | 독립 설정(키 로드, 임계값 0.7m, N_REPEAT) | — |

## ⚠ 데이터 현황 (중요)

- 현재 `data/`에는 **저장된 이미지 프레임이 없고**, `dist_log*.csv`는 **30cm 단일 조건**입니다.
- VLM-only 베이스라인은 이미지가 있어야 돌아갑니다 → **새 데이터 수집이 필요**합니다.
- 지금 당장 가능한 것: `run_ours_from_log.py`로 Ours 거리 **반복정밀도(std)** 확인 + 지표 파이프라인 검증.

## 실행 순서 (전체 비교)

```bash
source ../.venv/bin/activate      # 루트 .venv 사용

# 0) 지표 모듈 검증
python metrics.py

# 1) 데이터 수집 — 거리 조건별로 반복 (near/boundary/far 균형 있게)
python collect_frames.py --distance 0.30 --label "cell phone" --shots 20
python collect_frames.py --distance 0.90 --label "cup"        --shots 20
python collect_frames.py --distance 1.30 --label "bottle"     --shots 20

# 2) VLM-only 베이스라인 실행 (같은 프레임)
python vlm_baseline.py --manifest data/manifest.csv

# 3) 본문 Ours 결과 자동 산출 (본문 수정 없음, import만)
python run_ours.py --manifest data/manifest.csv

# 4) 비교표 + 그림 생성
python compare.py --manifest data/manifest.csv \
                  --vlm results/vlm_results.csv \
                  --ours results/ours_results.csv
```

산출물은 `results/`:
- `main_table.md` / `main_table.csv` — 논문 Main Table
- `bland_altman.png` — 방법별 계통오차
- `stratified_mae.png` — near/boundary/far 구간별 오차 (경계 구간이 승부처)

## 지표 요약

**거리**: MAE, RMSE, AbsRel, bias, δ<1.25, Pearson r, 예측 std(반복정밀도)
**결정(실행가능)**: Accuracy, Precision/Recall/F1, **False-Trigger Rate**(오발동), Cohen's κ
**안정성**: **flip-rate**(같은 프레임 N회 질의 시 결정 뒤집힘) — 기하는 결정적이라 0
**통계검정**: 거리오차 Wilcoxon signed-rank, 결정 McNemar 정확검정, MAE 부트스트랩 95%CI

## 리뷰어 방어 포인트

- "검출기 차이 아니냐" → §통제변인: **동일 YOLO 박스**를 양쪽에 제공(`manifest`).
- "프롬프트를 더 잘 짜면?" → `vlm_baseline.py`의 프롬프트를 변형해 best-VLM과도 비교.
- "GT 오차는?" → `collect_frames`의 실측 거리(줄자/레이저) 오차를 명기.
