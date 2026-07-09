"""SAM/Depth Anything 캐시 interval 1~10 정확도 비교 실험."""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test3"
FRAME_RESULT = RESULT_DIR / "test3_frame_results.csv"
SUMMARY_RESULT = RESULT_DIR / "test3_interval_summary.csv"
TEXT_RESULT = RESULT_DIR / "test3_res.txt"
METADATA_RESULT = RESULT_DIR / "test3_metadata.json"

CAPTURE_SECONDS = 10.0
SAMPLE_FPS = 5.0
INTERVALS = range(1, 11)
BBOX_IOU_THRESHOLD = 0.7


@dataclass
class FreshFrameResult:
    frame_index: int
    timestamp_seconds: float
    labels: list[str]
    bboxes: list[list[float]]
    mask_count: int
    union_mask: np.ndarray
    raw_depth: np.ndarray
    yolo_seconds: float
    sam_seconds: float
    depth_seconds: float


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def cache_is_compatible(
    current: FreshFrameResult,
    previous: FreshFrameResult | None,
    cached: FreshFrameResult,
) -> bool:
    """서버처럼 현재 객체와 직전 처리 프레임의 라벨/bbox를 비교한다."""
    return (
        previous is not None
        and bool(cached.labels)
        and cached.mask_count == len(current.labels)
        and current.labels == previous.labels
        and len(current.bboxes) == len(previous.bboxes)
        and all(
            bbox_iou(previous_bbox, current_bbox) >= BBOX_IOU_THRESHOLD
            for previous_bbox, current_bbox in zip(previous.bboxes, current.bboxes)
        )
    )


def union_masks(masks: list[np.ndarray], frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    combined = np.zeros((height, width), dtype=bool)
    for mask in masks:
        current = mask
        if current.shape != combined.shape:
            current = cv2.resize(
                current.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        combined |= current
    return combined


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=np.float32)
    minimum = float(depth[finite].min())
    maximum = float(depth[finite].max())
    normalized = np.zeros_like(depth, dtype=np.float32)
    if maximum - minimum > 1e-6:
        normalized[finite] = (depth[finite] - minimum) / (maximum - minimum)
    return normalized


def sam_difference_ratio(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """두 union mask의 차이를 1 - IoU로 계산한다(0=동일, 1=완전 불일치)."""
    intersection = np.logical_and(baseline, candidate).sum(dtype=np.int64)
    union = np.logical_or(baseline, candidate).sum(dtype=np.int64)
    return 0.0 if union == 0 else float(1.0 - intersection / union)


def depth_difference_ratio(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """0~1 정규화 depth map 사이의 평균 절대오차를 반환한다."""
    if baseline.shape != candidate.shape:
        candidate = cv2.resize(
            candidate,
            (baseline.shape[1], baseline.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return float(np.mean(np.abs(baseline - candidate)))


def metric_depth_mae(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """Metric Depth map의 평균 절대오차를 미터 단위로 반환한다."""
    if baseline.shape != candidate.shape:
        candidate = cv2.resize(
            candidate,
            (baseline.shape[1], baseline.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    finite = np.isfinite(baseline) & np.isfinite(candidate)
    if not finite.any():
        return 0.0
    return float(np.mean(np.abs(baseline[finite] - candidate[finite])))


def metric_depth_relative_error(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """Metric Depth MAE를 기준 프레임의 평균 깊이로 나눈 상대 오차를 반환한다."""
    finite = np.isfinite(baseline)
    denominator = float(np.mean(np.abs(baseline[finite]))) if finite.any() else 0.0
    if denominator <= 1e-6:
        return 0.0
    return metric_depth_mae(baseline, candidate) / denominator


def capture_frames() -> tuple[list[np.ndarray], list[float]]:
    from vision.stream import WebcamStream

    stream = WebcamStream()
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    started_at = time.perf_counter()
    next_sample_at = 0.0
    sample_period = 1.0 / SAMPLE_FPS

    print(f"[test3] {CAPTURE_SECONDS:.0f}초 촬영을 시작합니다. 자연스럽게 움직여 주세요.")
    try:
        while True:
            ret, frame = stream.get_frame()
            if not ret:
                continue
            elapsed = time.perf_counter() - started_at
            if elapsed >= CAPTURE_SECONDS:
                break
            if elapsed >= next_sample_at:
                frames.append(frame.copy())
                timestamps.append(elapsed)
                next_sample_at += sample_period

            remaining = max(0.0, CAPTURE_SECONDS - elapsed)
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Recording: {remaining:04.1f}s / samples: {len(frames)}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            cv2.imshow("test3 - 10 second capture", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
    finally:
        stream.release()

    if not frames:
        raise RuntimeError("촬영된 프레임이 없습니다.")
    print(f"[test3] 촬영 완료: {len(frames)}개 프레임")
    return frames, timestamps


def compute_fresh_results(
    frames: list[np.ndarray], timestamps: list[float]
) -> list[FreshFrameResult]:
    from vision.depth.depth_estimator import DepthEstimator
    from vision.detector import ObjectDetector
    from vision.segmentation.segmenter import ObjectSegmenter

    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth_estimator = DepthEstimator()
    results: list[FreshFrameResult] = []

    print("[test3] interval=1 기준값 계산: 모든 프레임에서 YOLO/SAM/Depth를 새로 실행합니다.")
    for index, (frame, captured_at) in enumerate(zip(frames, timestamps), start=1):
        started = time.perf_counter()
        yolo_result = detector.detect(frame)
        scene = detector.build_scene(yolo_result, frame, index)
        yolo_seconds = time.perf_counter() - started

        started = time.perf_counter()
        annotated, scene, masks = segmenter.segment_objects(frame, scene)
        sam_seconds = time.perf_counter() - started

        started = time.perf_counter()
        depth_data = depth_estimator.get_depth_map(frame)
        depth_seconds = time.perf_counter() - started

        labels = [obj["label"] for obj in scene.get("objects", [])]
        bboxes = [obj["yolo"]["bbox_2d"] for obj in scene.get("objects", [])]
        results.append(
            FreshFrameResult(
                frame_index=index,
                timestamp_seconds=captured_at,
                labels=labels,
                bboxes=bboxes,
                mask_count=len(masks),
                union_mask=union_masks(masks, frame.shape),
                raw_depth=np.asarray(depth_data["raw_depth"], dtype=np.float32),
                yolo_seconds=yolo_seconds,
                sam_seconds=sam_seconds,
                depth_seconds=depth_seconds,
            )
        )

        cv2.putText(
            annotated,
            f"Baseline {index}/{len(frames)}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        cv2.imshow("test3 - interval 1 baseline", annotated)
        cv2.waitKey(1)
        print(
            f"[test3] 기준 프레임 {index}/{len(frames)} | objects={len(labels)} | "
            f"YOLO={yolo_seconds:.3f}s SAM={sam_seconds:.3f}s Depth={depth_seconds:.3f}s"
        )

    cv2.destroyAllWindows()
    return results


def evaluate_intervals(fresh_results: list[FreshFrameResult]) -> list[dict]:
    rows: list[dict] = []
    for interval in INTERVALS:
        cached: FreshFrameResult | None = None
        previous: FreshFrameResult | None = None
        for current in fresh_results:
            # server의 sam_frame_count는 1부터 시작하며 N의 배수에서 갱신한다.
            scheduled_refresh = current.frame_index % interval == 0
            compatible = (
                cached is not None
                and cache_is_compatible(current, previous, cached)
            )
            refresh = cached is None or scheduled_refresh or not compatible
            if cached is None:
                refresh_reason = "initial"
            elif scheduled_refresh:
                refresh_reason = "interval"
            elif not compatible:
                refresh_reason = "cache_invalid"
            else:
                refresh_reason = "cache"
            if refresh:
                cached = current

            assert cached is not None
            current_normalized_depth = normalize_depth(current.raw_depth)
            cached_normalized_depth = normalize_depth(cached.raw_depth)
            rows.append(
                {
                    "interval": interval,
                    "frame_index": current.frame_index,
                    "timestamp_seconds": current.timestamp_seconds,
                    "used_cache": not refresh,
                    "refresh_reason": refresh_reason,
                    "object_count": len(current.labels),
                    "labels": "|".join(current.labels),
                    "sam_difference_ratio": sam_difference_ratio(
                        current.union_mask, cached.union_mask
                    ),
                    "depth_difference_ratio": depth_difference_ratio(
                        current_normalized_depth, cached_normalized_depth
                    ),
                    "metric_depth_mae_m": metric_depth_mae(
                        current.raw_depth, cached.raw_depth
                    ),
                    "metric_depth_relative_error": metric_depth_relative_error(
                        current.raw_depth, cached.raw_depth
                    ),
                }
            )
            # 서버도 캐시 재사용 여부와 무관하게 매 처리 프레임의 bbox/label을 저장한다.
            previous = current
    return rows


def save_results(fresh_results: list[FreshFrameResult], rows: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame_fields = list(rows[0].keys())
    with FRAME_RESULT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for interval in INTERVALS:
        selected = [row for row in rows if row["interval"] == interval]
        sam_values = np.array([row["sam_difference_ratio"] for row in selected])
        depth_values = np.array([row["depth_difference_ratio"] for row in selected])
        metric_mae_values = np.array([row["metric_depth_mae_m"] for row in selected])
        metric_relative_values = np.array(
            [row["metric_depth_relative_error"] for row in selected]
        )
        cache_count = sum(bool(row["used_cache"]) for row in selected)
        summary_rows.append(
            {
                "interval": interval,
                "frame_count": len(selected),
                "cache_count": cache_count,
                "cache_rate": cache_count / len(selected),
                "mean_sam_difference_ratio": float(sam_values.mean()),
                "max_sam_difference_ratio": float(sam_values.max()),
                "mean_depth_difference_ratio": float(depth_values.mean()),
                "max_depth_difference_ratio": float(depth_values.max()),
                "mean_metric_depth_mae_m": float(metric_mae_values.mean()),
                "max_metric_depth_mae_m": float(metric_mae_values.max()),
                "mean_metric_depth_relative_error": float(metric_relative_values.mean()),
                "max_metric_depth_relative_error": float(metric_relative_values.max()),
            }
        )

    with SUMMARY_RESULT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with TEXT_RESULT.open("w", encoding="utf-8") as file:
        file.write("test3: SAM/Depth cache interval experiment\n")
        file.write(f"created_at: {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        file.write(f"capture_seconds: {CAPTURE_SECONDS}\n")
        file.write(f"sample_fps: {SAMPLE_FPS}\n")
        file.write(f"frame_count: {len(fresh_results)}\n\n")
        file.write("[interval summary]\n")
        for summary in summary_rows:
            file.write(
                f"interval={summary['interval']:2d} | "
                f"cache_rate={summary['cache_rate']:.6f} | "
                f"sam_mean={summary['mean_sam_difference_ratio']:.6f} | "
                f"sam_max={summary['max_sam_difference_ratio']:.6f} | "
                f"depth_mean={summary['mean_depth_difference_ratio']:.6f} | "
                f"depth_max={summary['max_depth_difference_ratio']:.6f} | "
                f"metric_mae_mean={summary['mean_metric_depth_mae_m']:.6f}m | "
                f"metric_mae_max={summary['max_metric_depth_mae_m']:.6f}m\n"
            )

        file.write("\n[frame results]\n")
        for row in rows:
            file.write(
                f"interval={row['interval']:2d} | "
                f"frame={row['frame_index']:3d} | "
                f"time={row['timestamp_seconds']:.3f}s | "
                f"used_cache={row['used_cache']} | "
                f"reason={row['refresh_reason']} | "
                f"sam_diff={row['sam_difference_ratio']:.6f} | "
                f"depth_diff={row['depth_difference_ratio']:.6f} | "
                f"metric_depth_mae={row['metric_depth_mae_m']:.6f}m | "
                f"metric_depth_relative={row['metric_depth_relative_error']:.6f} | "
                f"labels={row['labels']}\n"
            )

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "capture_seconds": CAPTURE_SECONDS,
        "sample_fps": SAMPLE_FPS,
        "captured_frame_count": len(fresh_results),
        "intervals": list(INTERVALS),
        "bbox_iou_threshold": BBOX_IOU_THRESHOLD,
        "sam_metric": "1 - IoU of baseline and cached union masks",
        "depth_metric": "MAE of independently min-max-normalized baseline and cached depth maps",
        "metric_depth_mae": "MAE in meters between baseline and cached Metric Depth maps",
        "metric_depth_relative_error": "Metric Depth MAE divided by baseline mean absolute depth",
        "baseline_mean_yolo_seconds": float(
            np.mean([result.yolo_seconds for result in fresh_results])
        ),
        "baseline_mean_sam_seconds": float(
            np.mean([result.sam_seconds for result in fresh_results])
        ),
        "baseline_mean_depth_seconds": float(
            np.mean([result.depth_seconds for result in fresh_results])
        ),
    }
    METADATA_RESULT.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        frames, timestamps = capture_frames()
        fresh_results = compute_fresh_results(frames, timestamps)
        rows = evaluate_intervals(fresh_results)
        save_results(fresh_results, rows)
    except KeyboardInterrupt:
        print("\n[test3] 사용자 요청으로 실험을 중단합니다.")
        return 1

    print(f"[test3] 실험 완료: {FRAME_RESULT}")
    print(f"[test3] interval 요약: {SUMMARY_RESULT}")
    print(f"[test3] 텍스트 결과: {TEXT_RESULT}")
    print(f"[test3] 메타데이터: {METADATA_RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
