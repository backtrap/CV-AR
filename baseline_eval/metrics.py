"""
지표 계산 모듈 (numpy 전용, scipy/pandas 미사용).

거리 회귀 지표, 실행가능성(이진 결정) 지표, 안정성(flip-rate),
그리고 짝지어진 표본용 통계 검정(McNemar, Wilcoxon signed-rank),
부트스트랩 신뢰구간, Bland-Altman 통계를 제공한다.

모든 함수는 순수 함수이며 외부 상태에 의존하지 않는다.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

_RNG = np.random.default_rng(42)  # 부트스트랩 재현성을 위한 고정 시드


# ---------------------------------------------------------------------------
# 거리 회귀 지표
# ---------------------------------------------------------------------------
def regression_metrics(pred: Sequence[float], gt: Sequence[float]) -> dict:
    """예측 거리 vs 실측 거리의 회귀 오차 지표.

    - mae, rmse: 절대/제곱평균 오차 (m)
    - absrel: 상대오차 평균 |pred-gt|/gt
    - bias: 부호 있는 평균오차 (pred-gt), 계통적 과대/과소추정 확인용
    - delta1: max(pred/gt, gt/pred) < 1.25 를 만족하는 비율 (깊이추정 관례 지표)
    - pearson_r: 상관계수 (단일 GT값만 있으면 nan)
    """
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    diff = pred - gt
    valid = gt > 1e-9

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    absrel = float(np.mean(np.abs(diff[valid]) / gt[valid])) if valid.any() else float("nan")
    bias = float(np.mean(diff))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(pred[valid] / gt[valid], gt[valid] / pred[valid])
    delta1 = float(np.mean(ratio < 1.25)) if valid.any() else float("nan")

    if np.std(pred) < 1e-12 or np.std(gt) < 1e-12:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(pred, gt)[0, 1])

    return {
        "n": int(len(pred)),
        "mae": mae,
        "rmse": rmse,
        "absrel": absrel,
        "bias": bias,
        "delta1": delta1,
        "pearson_r": pearson_r,
        "pred_std": float(np.std(pred)),  # 단일조건 반복성(정밀도) 지표
    }


# ---------------------------------------------------------------------------
# 실행가능성(이진 결정) 지표
# ---------------------------------------------------------------------------
def classification_metrics(pred: Sequence[int], gt: Sequence[int]) -> dict:
    """실행가능(1)/불가(0) 결정의 정확도 지표.

    false_trigger_rate: 실제로 닿지 못하는데(gt=0) 실행가능으로 오판한 비율
                        = FP / (FP + TN). PPS 오발동을 직접 반영하는 핵심 지표.
    """
    pred = np.asarray(pred, dtype=int)
    gt = np.asarray(gt, dtype=int)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    acc = (tp + tn) / max(1, len(gt))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    false_trigger = fp / max(1, fp + tn)
    kappa = _cohen_kappa(pred, gt)

    return {
        "n": int(len(gt)),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_trigger_rate": false_trigger,
        "cohen_kappa": kappa,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _cohen_kappa(pred: np.ndarray, gt: np.ndarray) -> float:
    n = len(gt)
    if n == 0:
        return float("nan")
    po = float(np.mean(pred == gt))
    p1 = np.mean(pred == 1)
    g1 = np.mean(gt == 1)
    pe = p1 * g1 + (1 - p1) * (1 - g1)
    return float((po - pe) / (1 - pe)) if (1 - pe) > 1e-12 else float("nan")


def flip_rate(decisions_per_item: Sequence[Sequence[int]]) -> dict:
    """항목별 반복 질의 결과에서 결정이 뒤집히는 비율.

    decisions_per_item: [[1,1,0,1,1], [0,0,0,0,0], ...] 형태.
    결정적(deterministic) 방식이면 모두 0 → mean_flip_rate=0.
    """
    rates = []
    for reps in decisions_per_item:
        reps = np.asarray(reps, dtype=int)
        if len(reps) <= 1:
            rates.append(0.0)
            continue
        majority = int(round(reps.mean()))
        rates.append(float(np.mean(reps != majority)))
    return {
        "mean_flip_rate": float(np.mean(rates)) if rates else 0.0,
        "n_unstable_items": int(np.sum(np.asarray(rates) > 0)),
        "per_item": rates,
    }


# ---------------------------------------------------------------------------
# 짝지어진 표본 통계 검정
# ---------------------------------------------------------------------------
def mcnemar_exact(correct_a: Sequence[int], correct_b: Sequence[int]) -> dict:
    """두 방법의 정오답(같은 표본)에 대한 McNemar 정확검정(이항).

    b = A정답·B오답, c = A오답·B정답. H0: 두 방법의 오류율 동일.
    두꼬리 정확 p-value를 이항분포로 계산한다.
    """
    a = np.asarray(correct_a, dtype=int)
    b_ = np.asarray(correct_b, dtype=int)
    b = int(np.sum((a == 1) & (b_ == 0)))
    c = int(np.sum((a == 0) & (b_ == 1)))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "p_value": float(p)}


def wilcoxon_signed_rank(err_a: Sequence[float], err_b: Sequence[float]) -> dict:
    """짝지어진 오차(err_a vs err_b)의 Wilcoxon signed-rank 검정.

    정규근사(연속성 보정 포함). scipy 없이 구현.
    H0: 두 방법의 오차 분포 차이의 중앙값이 0.
    """
    a = np.asarray(err_a, dtype=float)
    b = np.asarray(err_b, dtype=float)
    d = a - b
    d = d[np.abs(d) > 1e-12]  # 0차이 제거
    n = len(d)
    if n < 1:
        return {"n": 0, "w_plus": float("nan"), "z": float("nan"), "p_value": float("nan")}

    ranks = _average_ranks(np.abs(d))
    w_plus = float(np.sum(ranks[d > 0]))
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w < 1e-12:
        return {"n": n, "w_plus": w_plus, "z": float("nan"), "p_value": float("nan")}
    cc = 0.5 * np.sign(w_plus - mean_w)  # 연속성 보정
    z = (w_plus - mean_w - cc) / math.sqrt(var_w)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return {"n": n, "w_plus": w_plus, "z": float(z), "p_value": float(min(1.0, p))}


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based 평균 순위
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# 부트스트랩 & Bland-Altman
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> dict:
    """통계량의 부트스트랩 백분위 신뢰구간 (기본 95%)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    boots = np.array([
        stat_fn(values[_RNG.integers(0, n, n)]) for _ in range(n_boot)
    ])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": float(stat_fn(values)), "lo": lo, "hi": hi}


def bland_altman(pred: Sequence[float], gt: Sequence[float]) -> dict:
    """Bland-Altman: 평균차(bias)와 일치한계(LoA = bias ± 1.96·SD)."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    diff = pred - gt
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "loa_lower": mean_diff - 1.96 * sd_diff,
        "loa_upper": mean_diff + 1.96 * sd_diff,
    }


def stratify_by_distance(gt: Sequence[float], near_max: float, boundary_max: float) -> dict:
    """GT 거리로 near/boundary/far 인덱스를 나눈다."""
    gt = np.asarray(gt, dtype=float)
    return {
        "near": np.where(gt < near_max)[0],
        "boundary": np.where((gt >= near_max) & (gt <= boundary_max))[0],
        "far": np.where(gt > boundary_max)[0],
    }


if __name__ == "__main__":
    # 스모크 테스트: 합성 데이터로 각 지표가 동작하는지 확인
    rng = np.random.default_rng(0)
    gt_d = rng.uniform(0.2, 2.0, 60)
    ours = gt_d + rng.normal(0, 0.05, 60)          # 정밀한 기하 추정 가정
    vlm = gt_d + rng.normal(0.1, 0.35, 60)          # 편향+큰 분산의 VLM 가정
    gt_exec = (gt_d <= 0.7).astype(int)
    ours_exec = (ours <= 0.7).astype(int)
    vlm_exec = (vlm <= 0.7).astype(int)

    print("[Ours 거리]", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in regression_metrics(ours, gt_d).items()})
    print("[VLM  거리]", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in regression_metrics(vlm, gt_d).items()})
    print("[Ours 결정]", classification_metrics(ours_exec, gt_exec))
    print("[VLM  결정]", classification_metrics(vlm_exec, gt_exec))
    print("[McNemar]", mcnemar_exact(ours_exec == gt_exec, vlm_exec == gt_exec))
    print("[Wilcoxon]", wilcoxon_signed_rank(np.abs(ours - gt_d), np.abs(vlm - gt_d)))
    print("[Bland-Altman VLM]", bland_altman(vlm, gt_d))
    print("✅ metrics.py 스모크 테스트 통과")
