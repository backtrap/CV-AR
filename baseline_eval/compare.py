"""
Ours(Depth+PPS) vs VLM-only 비교표/그림 생성.

입력:
  --manifest  GT (frame_id,label,gt_distance_m,gt_executable, ...)
  --vlm       results/vlm_results.csv  (vlm_baseline.py 산출)
  --ours      results/ours_results.csv (frame_id,ours_distance_m[,ours_executable])
              ours_executable가 없으면 PPS 임계값으로 유도한다.

출력:
  results/main_table.md   논문 Main Table (마크다운)
  results/main_table.csv
  results/bland_altman.png (matplotlib 사용 가능 시)

frame_id를 키로 세 소스를 inner-join 하여 '같은 프레임'에서만 비교한다(paired).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import metrics as M
from config import PPS_THRESHOLD_M, NEAR_MAX, BOUNDARY_MAX, DIR, RESULTS_DIR


def _read(path: Path) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    return {r["frame_id"]: r for r in csv.DictReader(open(path, newline=""))}


def _exec_from_dist(dist: float) -> int:
    return int(dist <= PPS_THRESHOLD_M)


def build(manifest: Path, vlm: Path, ours: Path, with_ablation: bool = False) -> None:
    gt = _read(manifest)
    vlm_r = _read(vlm)
    ours_r = _read(ours)
    have_vlm = bool(vlm_r)
    have_ours = bool(ours_r)

    ksets = [set(gt)]
    if have_vlm:
        ksets.append(set(vlm_r))
    if have_ours:
        ksets.append(set(ours_r))
    keys = sorted(set.intersection(*ksets), key=lambda x: int(x) if x.isdigit() else x)
    if not keys:
        print("공통 frame_id 없음 — manifest/vlm/ours의 frame_id 정합을 확인하세요.")
        print(f"  manifest={len(gt)} vlm={len(vlm_r)} ours={len(ours_r)}")
        return

    gt_d, gt_e, labels = [], [], []
    vlm_d, vlm_e, vlm_flip = [], [], []
    ours_d, ours_e = [], []

    for k in keys:
        gd = float(gt[k]["gt_distance_m"])
        gt_d.append(gd)
        gt_e.append(int(float(gt[k].get("gt_executable", _exec_from_dist(gd)))))
        labels.append(gt[k].get("label", ""))

        if have_vlm:
            vlm_d.append(float(vlm_r[k]["vlm_distance_m"]))
            vlm_e.append(int(float(vlm_r[k]["vlm_executable"])))
            vlm_flip.append(float(vlm_r[k].get("flip_rate", 0.0)))

        if have_ours:
            od = float(ours_r[k]["ours_distance_m"])
            ours_d.append(od)
            oe = ours_r[k].get("ours_executable")
            ours_e.append(int(float(oe)) if oe not in (None, "") else _exec_from_dist(od))

    gt_d = np.array(gt_d)
    rows = []
    abl_e = _ablation_exec(labels) if with_ablation else None
    if abl_e is not None:
        rows.append(_method_row_decision("Ours − geometry (static)", abl_e, gt_e))
    if have_vlm:
        rows.append(_method_row("VLM-only (GPT-4V)", vlm_d, vlm_e, gt_d, gt_e, vlm_flip))
    if have_ours:
        rows.append(_method_row("Ours (Depth+PPS)", ours_d, ours_e, gt_d, gt_e,
                                [0.0] * len(keys)))  # 결정적 → flip 0

    _write_table(rows, keys)
    _stats(ours_d, vlm_d, ours_e, vlm_e, gt_d, gt_e, have_ours, have_vlm, abl_e)
    if have_ours and have_vlm:
        _plot_bland_altman(ours_d, vlm_d, gt_d, True)
        _plot_stratified(gt_d, ours_d, vlm_d, True)


def _ablation_exec(labels):
    """Self-ablation 'Ours − 기하': baseline_eval 내부 독립 모듈(ablation.py) 사용.
    본문을 import하지 않는다(레지스트리는 ablation.py에 미러링됨).
    """
    from ablation import ablation_executable
    return ablation_executable(labels)


def _method_row(name, pred_d, pred_e, gt_d, gt_e, flips):
    reg = M.regression_metrics(pred_d, gt_d)
    cls = M.classification_metrics(pred_e, gt_e)
    return {
        "method": name,
        "mae": reg["mae"], "absrel": reg["absrel"], "delta1": reg["delta1"],
        "f1": cls["f1"], "false_trigger": cls["false_trigger_rate"],
        "kappa": cls["cohen_kappa"],
        "flip": float(np.mean(flips)),
    }


def _method_row_decision(name, pred_e, gt_e):
    """거리 출력이 없는 방법(정적 ablation)용 — 결정 지표만, 거리 열은 N/A."""
    cls = M.classification_metrics(pred_e, gt_e)
    nan = float("nan")
    return {
        "method": name,
        "mae": nan, "absrel": nan, "delta1": nan,
        "f1": cls["f1"], "false_trigger": cls["false_trigger_rate"],
        "kappa": cls["cohen_kappa"], "flip": 0.0,
    }


def _write_table(rows, keys):
    hdr = ["method", "mae", "absrel", "delta1", "f1", "false_trigger", "kappa", "flip"]
    label = {"method": "Method", "mae": "Dist.MAE↓(m)", "absrel": "AbsRel↓",
             "delta1": "δ<1.25↑", "f1": "Decision F1↑", "false_trigger": "FalseTrig↓",
             "kappa": "κ↑", "flip": "Flip%↓"}
    lines = [f"# Baseline 비교 결과 (n={len(keys)} paired frames)\n",
             "| " + " | ".join(label[h] for h in hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    def _fmt(v):
        return "—" if isinstance(v, float) and v != v else f"{v:.4f}"  # nan → —
    for r in rows:
        cells = [r["method"]] + [_fmt(r[h]) for h in hdr[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / "main_table.md").write_text(md, encoding="utf-8")
    with open(RESULTS_DIR / "main_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print(md)
    print(f"저장: {RESULTS_DIR/'main_table.md'}, main_table.csv")


def _stats(ours_d, vlm_d, ours_e, vlm_e, gt_d, gt_e, have_ours, have_vlm, abl_e=None):
    print("\n[통계검정 · 같은 프레임 paired]")
    if have_ours and have_vlm:
        err_o = np.abs(np.array(ours_d) - gt_d)
        err_v = np.abs(np.array(vlm_d) - gt_d)
        wil = M.wilcoxon_signed_rank(err_o, err_v)
        mc = M.mcnemar_exact(np.array(ours_e) == np.array(gt_e),
                             np.array(vlm_e) == np.array(gt_e))
        ci_o = M.bootstrap_ci(err_o)
        ci_v = M.bootstrap_ci(err_v)
        print(f"  거리오차 Wilcoxon (Ours vs VLM): z={wil['z']:.3f}, p={wil['p_value']:.3e} "
              f"(Ours가 유의하게 낮으면 p<0.05)")
        print(f"  결정 McNemar (Ours vs VLM): b={mc['b']} c={mc['c']}, p={mc['p_value']:.3e}")
        print(f"  거리 MAE 95%CI  Ours=[{ci_o['lo']:.3f},{ci_o['hi']:.3f}] "
              f"VLM=[{ci_v['lo']:.3f},{ci_v['hi']:.3f}]")

    if abl_e is not None and have_ours:
        mc2 = M.mcnemar_exact(np.array(ours_e) == np.array(gt_e),
                              np.array(abl_e) == np.array(gt_e))
        print(f"  ★ 결정 McNemar (Ours vs −기하 ablation): b={mc2['b']} c={mc2['c']}, "
              f"p={mc2['p_value']:.3e}")
        print("    → p<0.05면 PPS 기하가 '정적 어포던스' 대비 유의하게 기여 (self-baseline 핵심)")
    elif not have_ours:
        print("  ours_results.csv 없음 — Ours 관련 검정 생략 (run_ours.py 먼저 실행)")


def _plot_bland_altman(ours_d, vlm_d, gt_d, have_ours):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib 불가 — 그림 생략 ({e})")
        return
    fig, axes = plt.subplots(1, 2 if have_ours else 1, figsize=(11, 4.5), squeeze=False)
    series = [("VLM-only", np.array(vlm_d))]
    if have_ours:
        series = [("Ours (Depth+PPS)", np.array(ours_d))] + series
    for ax, (name, pred) in zip(axes[0], series):
        ba = M.bland_altman(pred, gt_d)
        mean_xy = (pred + gt_d) / 2
        diff = pred - gt_d
        ax.scatter(mean_xy, diff, s=18, alpha=0.6)
        ax.axhline(ba["mean_diff"], color="k", ls="-", lw=1, label=f"bias={ba['mean_diff']:.3f}")
        ax.axhline(ba["loa_upper"], color="r", ls="--", lw=1, label="±1.96·SD")
        ax.axhline(ba["loa_lower"], color="r", ls="--", lw=1)
        ax.set_title(f"Bland-Altman: {name}")
        ax.set_xlabel("(pred+gt)/2  (m)")
        ax.set_ylabel("pred - gt  (m)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "bland_altman.png"
    fig.savefig(out, dpi=130)
    print(f"저장: {out}")


def _plot_stratified(gt_d, ours_d, vlm_d, have_ours):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    strata = M.stratify_by_distance(gt_d, NEAR_MAX, BOUNDARY_MAX)
    names = ["near\n(<0.7)", "boundary\n(0.7~1.2)", "far\n(>1.2)"]
    keys = ["near", "boundary", "far"]
    vlm_mae = [_mae_idx(vlm_d, gt_d, strata[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3)
    w = 0.38
    ax.bar(x - (w/2 if have_ours else 0), vlm_mae, w if have_ours else 0.6, label="VLM-only")
    if have_ours:
        ours_mae = [_mae_idx(ours_d, gt_d, strata[k]) for k in keys]
        ax.bar(x + w/2, ours_mae, w, label="Ours (Depth+PPS)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Distance MAE (m)")
    ax.set_title("MAE by distance range (boundary = decisive)")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "stratified_mae.png"
    fig.savefig(out, dpi=130)
    print(f"저장: {out}")


def _mae_idx(pred, gt, idx):
    if len(idx) == 0:
        return 0.0
    pred = np.asarray(pred)
    return float(np.mean(np.abs(pred[idx] - gt[idx])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ours vs VLM 비교표/그림 생성")
    ap.add_argument("--manifest", default=str(DIR / "data" / "manifest.csv"))
    ap.add_argument("--vlm", default=str(RESULTS_DIR / "vlm_results.csv"))
    ap.add_argument("--ours", default=str(RESULTS_DIR / "ours_results.csv"))
    ap.add_argument("--with_ablation", action="store_true",
                    help="'Ours − 기하(정적 어포던스)' self-ablation 행 추가")
    a = ap.parse_args()
    build(Path(a.manifest), Path(a.vlm), Path(a.ours), with_ablation=a.with_ablation)
