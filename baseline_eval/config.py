"""
baseline_eval 독립 설정.

본문(llm/, vision/)을 import 하지 않는다. 프로젝트 루트의 .env에서
OPENAI_API_KEY만 읽어 쓴다(수정하지 않음).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # 프로젝트 루트
DIR = Path(__file__).resolve().parent           # baseline_eval/
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# 베이스라인은 시각 추론이 필요하므로 vision 지원 모델을 쓴다(기본 gpt-4o).
VLM_MODEL = os.environ.get("BASELINE_VLM_MODEL", "gpt-4o")

# 실행가능성(PPS) 판정 임계값 — 본문 affordance_engine 기준(0.7m)과 동일.
PPS_THRESHOLD_M = float(os.environ.get("BASELINE_PPS_THRESHOLD", "0.7"))

# VLM 안정성(flip-rate) 측정을 위한 동일 프레임 반복 질의 횟수.
N_REPEAT = int(os.environ.get("BASELINE_N_REPEAT", "5"))

# 거리 구간 층화 경계 (m). near < 0.7 ≤ boundary ≤ 1.2 < far
NEAR_MAX = 0.7
BOUNDARY_MAX = 1.2

FRAMES_DIR = DIR / "data" / "frames"
RESULTS_DIR = DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
