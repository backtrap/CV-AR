"""
기존 dist_log*.csv 로 'Ours(기하) 거리 추정'의 정확도/정밀도를 산출한다.
(이미지가 없어도 지금 바로 실행 가능한 분석 — 지표 파이프라인 검증용)

dist_log 스키마: frame,label,condition,raw_depth,agent_z,target_z,x_diff,z_diff,dist_ground
  - condition: 실측 거리 라벨(예 "30cm")  → GT
  - dist_ground: 시스템이 계산한 거리      → Ours 예측

주의: 현재 로그의 dist_ground는 pseudo-unit일 수 있다(예: 30cm인데 ~2.5).
      --scale 로 pseudo→m 환산 계수를 주면 보정 후 지표도 함께 본다.
      단일 condition(30cm)만 있으면 절대 MAE보다 '반복 정밀도(std)'가 의미 있다.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

import metrics as M
from config import DIR, RESULTS_DIR


def parse_condition_m(cond: str) -> float | None:
    """'30cm'->0.30, '0.5m'->0.5, '70'->0.70(cm 가정) 등 → meter."""
    s = cond.strip().lower()
    m = re.search(r"(\d+\.?\d*)", s)
    if not m:
        return None
    v = float(m.group(1))
    if "cm" in s:
        return v / 100.0
    if "m" in s:
        return v
    return v / 100.0  # 단위 없으면 cm 가정


def run(patterns: list[str], scale: float, label_filter: str | None) -> None:
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        print(f"파일 없음: {patterns}")
        return

    gt, pred, labels = [], [], []
    for fp in files:
        with open(fp, newline="") as f:
            for row in csv.DictReader(f):
                if label_filter and row.get("label") != label_filter:
                    continue
                gm = parse_condition_m(row.get("condition", ""))
                dg = row.get("dist_ground")
                if gm is None or dg in (None, "", "dist_ground"):
                    continue
                try:
                    gt.append(gm)
                    pred.append(float(dg) * scale)
                    labels.append(row.get("label"))
                except ValueError:
                    continue

    if not gt:
        print("유효 샘플 없음.")
        return

    gt = np.array(gt); pred = np.array(pred)
    reg = M.regression_metrics(pred, gt)
    ci = M.bootstrap_ci(np.abs(pred - gt))
    ba = M.bland_altman(pred, gt)

    n_cond = len(set(np.round(gt, 3)))
    report = {
        "files": [Path(f).name for f in files],
        "n_samples": int(len(gt)),
        "n_conditions": n_cond,
        "scale_applied": scale,
        "label_filter": label_filter,
        "regression": reg,
        "abs_err_mae_95ci": ci,
        "bland_altman": ba,
    }
    (RESULTS_DIR / "ours_from_log.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"--- Ours 기하 거리 분석 (n={len(gt)}, conditions={n_cond}, scale={scale}) ---")
    print(f"  MAE={reg['mae']:.4f} m | RMSE={reg['rmse']:.4f} | AbsRel={reg['absrel']:.4f}")
    print(f"  bias={reg['bias']:+.4f} m | δ<1.25={reg['delta1']:.3f}")
    print(f"  예측 표준편차(정밀도/반복성)={reg['pred_std']:.4f} m")
    print(f"  |err| MAE 95%CI=[{ci['lo']:.4f}, {ci['hi']:.4f}]")
    if n_cond == 1:
        print("  ⚠ 단일 거리 조건 → 절대정확도(MAE)보다 '표준편차=반복정밀도'가 핵심 지표.")
    if reg["absrel"] > 1.0:
        print("  ⚠ AbsRel이 매우 큼 → dist_ground가 pseudo-unit일 가능성. "
              "--scale 로 환산 계수를 주고 재실행하세요.")
    print(f"저장: {RESULTS_DIR/'ours_from_log.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="기존 dist_log로 Ours 거리 정확도 산출")
    ap.add_argument("--glob", nargs="+",
                    default=[str(DIR.parent / "data" / "dist_log*.csv")])
    ap.add_argument("--scale", type=float, default=1.0,
                    help="dist_ground(pseudo) → meter 환산 계수")
    ap.add_argument("--label", default=None, help="특정 라벨만 필터 (예: 'cell phone')")
    a = ap.parse_args()
    run(a.glob, a.scale, a.label)
