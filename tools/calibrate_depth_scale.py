"""
[Task 6] Depth 스케일 계수 근거화 도구

문제:
  기존 DEPTH_SCALE_FACTOR=0.51 은 단일 측정(실제 22cm / 화면 43cm)에서 나온
  '스케일만' 있는 임시값이다. metric depth 오차는 일반적으로 스케일(gain)뿐 아니라
  오프셋(bias)도 가지므로(true = a*pred + b), 여러 거리의 측정으로 회귀해야 근거가 생긴다.

사용법:
  1) 데이터 수집: 알려진 거리(예: 0.5/1.0/1.5/2.0/2.5 m)에 정지 객체를 두고
     각 거리에서 N프레임의 '모델 원시 예측 깊이'(pred, 보정 전)를 기록한다.
     - 원시 예측을 얻으려면 depth_scale.json 을 임시로 scale=1.0, offset=0.0 으로 두고
       객체 마스크의 robust_representative_depth(...) 값을 로깅한다.
  2) CSV 작성: 헤더 `true_distance_m,pred_depth_m` 로 (참값, 원시예측) 쌍을 채운다.
  3) 실행:  python -m tools.calibrate_depth_scale measurements.csv
     --write 를 주면 vision/depth/depth_scale.json 을 회귀 결과로 갱신한다.

출력:
  - scale-only 및 scale+offset 두 모델의 회귀식, R², RMSE(m), 거리별 잔차
  - 권장 모델(RMSE 작은 쪽)
"""

import argparse
import csv
import json
import os

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "vision", "depth", "depth_scale.json")


def _read_pairs(csv_path):
    true_v, pred_v = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "true_distance_m" not in reader.fieldnames \
                or "pred_depth_m" not in reader.fieldnames:
            raise ValueError("CSV 헤더에 true_distance_m, pred_depth_m 이 필요합니다.")
        for row in reader:
            try:
                t = float(row["true_distance_m"])
                p = float(row["pred_depth_m"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(t) and np.isfinite(p):
                true_v.append(t)
                pred_v.append(p)
    return np.asarray(true_v), np.asarray(pred_v)


def _stats(true_v, pred_hat):
    resid = true_v - pred_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((true_v - np.mean(true_v)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r2, resid


def calibrate(csv_path):
    true_v, pred_v = _read_pairs(csv_path)
    n = true_v.size
    if n < 2:
        raise ValueError(f"표본이 부족합니다(n={n}). 최소 2개 거리 이상 필요.")

    # 모델 A: scale-only (true = s*pred), 최소제곱 s = Σ(t·p)/Σ(p²)
    s_only = float(np.sum(true_v * pred_v) / np.sum(pred_v ** 2))
    rmse_a, r2_a, _ = _stats(true_v, s_only * pred_v)

    # 모델 B: scale+offset (true = a*pred + b), 1차 회귀
    a, b = np.polyfit(pred_v, true_v, 1)
    rmse_b, r2_b, resid_b = _stats(true_v, a * pred_v + b)

    print(f"표본 수 n = {n}")
    print(f"[모델 A] true = {s_only:.4f} * pred            | R²={r2_a:.4f} RMSE={rmse_a*100:.2f}cm")
    print(f"[모델 B] true = {a:.4f} * pred + {b:.4f}       | R²={r2_b:.4f} RMSE={rmse_b*100:.2f}cm")

    # 거리별 잔차(모델 B 기준)
    print("\n거리별 잔차(모델 B, cm):")
    order = np.argsort(true_v)
    for i in order:
        print(f"  참값 {true_v[i]:.2f}m  예측 {pred_v[i]:.3f}  보정후 {(a*pred_v[i]+b):.3f}  잔차 {resid_b[i]*100:+.1f}cm")

    use_offset = rmse_b < rmse_a
    chosen = {
        "model": "true_m = scale * pred_m + offset",
        "scale": round(a if use_offset else s_only, 6),
        "offset": round(b if use_offset else 0.0, 6),
        "unit": "meter",
        "provenance": f"{csv_path} 다거리 회귀 ({'scale+offset' if use_offset else 'scale-only'} 채택)",
        "n_samples": int(n),
        "r2": round(r2_b if use_offset else r2_a, 4),
        "rmse_m": round(rmse_b if use_offset else rmse_a, 4),
        "calibrated_at": None,  # Date.now 미사용 환경: 필요 시 수동 기입
        "note": "tools/calibrate_depth_scale.py 자동 생성.",
    }
    print(f"\n권장: {'모델 B(scale+offset)' if use_offset else '모델 A(scale-only)'}")
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Depth 스케일 계수 근거화(다거리 회귀)")
    ap.add_argument("csv", help="true_distance_m,pred_depth_m 헤더의 측정 CSV")
    ap.add_argument("--write", action="store_true", help="vision/depth/depth_scale.json 갱신")
    args = ap.parse_args()

    chosen = calibrate(args.csv)
    if args.write:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(chosen, f, ensure_ascii=False, indent=2)
        print(f"\n✅ {CONFIG_PATH} 갱신 완료")
    else:
        print("\n(미리보기) --write 를 주면 depth_scale.json 을 갱신합니다.")
        print(json.dumps(chosen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
