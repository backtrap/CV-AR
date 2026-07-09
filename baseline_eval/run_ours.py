"""
본문 Ours 파이프라인을 '수정 없이 import만' 하여, manifest의 각 프레임에서
person(agent)↔target 객체의 지면거리(ground-plane distance)를 계산한다.
→ results/ours_results.csv 생성 (compare.py 입력)

서버(llm/server_websocket.py)와 동일한 계산 경로를 재현한다:
  YOLO detect → SAM segment → Depth Anything V2 → 3D 역투영(캘리브레이션 행렬)
  → 지면거리 = sqrt((px-tx)^2 + (pz-tz)^2)   (relation_graph와 동일 정의)
  → ours_executable = (거리 <= PPS 0.7m)

공정성: manifest의 agent_bbox/object_bbox(= VLM에 준 것과 동일한 박스)와
        Ours가 검출한 박스를 IoU로 매칭해 '같은 객체'만 비교한다.

주의: 본문 모델 로딩을 위해 프로젝트 루트로 cwd를 옮긴다
      (YOLO("yolov8n.pt"), SAM("sam_b.pt")가 상대경로이기 때문).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

# baseline_eval 자체 설정 (본문 미수정)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PPS_THRESHOLD_M, DIR, RESULTS_DIR  # noqa: E402

ROOT = DIR.parent
sys.path.insert(0, str(ROOT))  # 본문 vision.* import 가능하도록 프로젝트 루트 추가


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _parse_box(s: str):
    return [float(v) for v in s.replace("|", " ").split()]


def _best_match(scene_objs, target_box, want_label=None):
    """manifest 박스와 IoU가 가장 큰 검출 객체를 반환."""
    best, best_iou = None, 0.0
    for obj in scene_objs:
        if obj is None or not obj.get("yolo"):
            continue
        if want_label is not None and obj["label"] != want_label:
            continue
        iou = _iou(obj["yolo"]["bbox_2d"], target_box)
        if iou > best_iou:
            best, best_iou = obj, iou
    return best, best_iou


# 캘리브레이션 기준 해상도 (vision/stream.py: 1280x720 고정 캡처 + 그 기준 CAMERA_MATRIX)
CALIB_W, CALIB_H = 1280, 720


def build_pipeline():
    """서버와 동일 구성으로 본문 모듈을 인스턴스화(수정 없음)."""
    from vision.stream import CAMERA_MATRIX
    from vision.detector import ObjectDetector
    from vision.segmentation.segmenter import ObjectSegmenter, SceneDepthAttacher
    from vision.depth.depth_estimator import DepthEstimator
    from vision.spatial.transformer import Spatial3DConverter

    return {
        "detector": ObjectDetector(),
        "segmenter": ObjectSegmenter(),
        "depth": DepthEstimator(),
        "attacher": SceneDepthAttacher(),
        "spatial": Spatial3DConverter(camera_matrix=CAMERA_MATRIX),
        # 720p 기준 원본 내부 파라미터 (프레임 해상도에 맞춰 매 프레임 스케일)
        "calib": (float(CAMERA_MATRIX[0, 0]), float(CAMERA_MATRIX[1, 1]),
                  float(CAMERA_MATRIX[0, 2]), float(CAMERA_MATRIX[1, 2])),
    }


def _scale_intrinsics(P, w, h):
    """프레임 해상도(w,h)에 맞춰 카메라 행렬을 스케일해 converter에 적용.

    캘리브레이션은 1280x720 기준이므로, 다른 해상도로 찍힌 프레임은
    fx,cx는 가로배율, fy,cy는 세로배율로 보정해야 3D 역투영이 미터로 정확하다.
    """
    sx, sy = w / CALIB_W, h / CALIB_H
    fx, fy, cx, cy = P["calib"]
    sp = P["spatial"]
    sp.f_x, sp.f_y = fx * sx, fy * sy
    sp.c_x, sp.c_y = cx * sx, cy * sy
    return abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3


def process_frame(P, frame, frame_id):
    """한 프레임 → spatial_3d까지 채워진 scene 반환 (서버 경로 재현)."""
    h, w = frame.shape[:2]
    _scale_intrinsics(P, w, h)  # 해상도-캘리브레이션 정합
    depth_data = P["depth"].get_depth_map(frame)
    yolo_result = P["detector"].detect(frame)
    scene = P["detector"].build_scene(yolo_result, frame, frame_id)
    _, scene, masks = P["segmenter"].segment_objects(frame, scene)
    scene = P["attacher"].attach_depth(scene, masks, depth_data)
    scene = P["spatial"].process_scene_3d(scene)
    return scene


def ground_distance(a, b) -> float:
    """relation_graph와 동일한 지면(XZ평면) 거리."""
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


def run(manifest_path: Path, out_path: Path, iou_min: float) -> None:
    import cv2

    os.chdir(ROOT)  # 상대경로 모델 로딩 위해 루트로 이동
    rows = list(csv.DictReader(open(manifest_path, newline="")))
    print(f"[run_ours] {len(rows)}개 프레임, 본문 파이프라인 로딩 중...")
    P = build_pipeline()

    results, skipped = [], 0
    for i, row in enumerate(rows):
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = DIR / img_path
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [skip] 이미지 없음: {img_path}")
            skipped += 1
            continue

        scene = process_frame(P, frame, int(row["frame_id"]) if row["frame_id"].isdigit() else i)
        objs = scene.get("objects", [])

        agent_box = _parse_box(row["agent_bbox"])
        target_box = _parse_box(row["object_bbox"])
        person, p_iou = _best_match(objs, agent_box, want_label="person")
        target, t_iou = _best_match(objs, target_box, want_label=row["label"])

        if person is None or target is None or p_iou < iou_min or t_iou < iou_min:
            print(f"  [skip] frame={row['frame_id']} 매칭실패 "
                  f"(person_iou={p_iou:.2f}, target_iou={t_iou:.2f})")
            skipped += 1
            continue

        ps, ts = person.get("spatial_3d"), target.get("spatial_3d")
        if not ps or not ts:
            print(f"  [skip] frame={row['frame_id']} spatial_3d 누락")
            skipped += 1
            continue

        dist = ground_distance(ps, ts)
        results.append({
            "frame_id": row["frame_id"],
            "ours_distance_m": round(dist, 4),
            "ours_executable": int(dist <= PPS_THRESHOLD_M),
            "person_z": round(ps["z"], 4),
            "target_z": round(ts["z"], 4),
            "match_iou": round(min(p_iou, t_iou), 3),
        })
        print(f"  [{i+1}/{len(rows)}] frame={row['frame_id']} "
              f"dist={dist:.3f}m exec={int(dist<=PPS_THRESHOLD_M)} iou={min(p_iou,t_iou):.2f}")

    if not results:
        print(f"결과 없음 (skipped={skipped}). 이미지/박스 매칭을 확인하세요.")
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"저장: {out_path} ({len(results)}행, skipped={skipped})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="본문 Ours로 프레임별 지면거리 산출")
    ap.add_argument("--manifest", default=str(DIR / "data" / "manifest.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "ours_results.csv"))
    ap.add_argument("--iou_min", type=float, default=0.3,
                    help="manifest 박스와 검출 박스의 최소 IoU (미달 시 skip)")
    a = ap.parse_args()
    run(Path(a.manifest), Path(a.out), a.iou_min)
