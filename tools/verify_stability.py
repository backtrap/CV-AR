"""
[Task 4 / §7 완료조건] 전체 파이프라인 라이브 좌표 안정성 검증.

전체 경로(웹캠→YOLO→SAM→Depth→3D→안정화)를 매 프레임 돌려서, 객체별 spatial_3d의
프레임 간 변동을 **안정화 전(raw_xyz) vs 후(stabilized)** 로 기록·비교한다.
정지 객체를 두고 돌리면 "객체별 3D 좌표가 안정적으로 생성됨"을 정량 입증할 수 있다.

주의: 정직한 지터 측정을 위해 Depth/SAM을 **매 프레임** 실행한다(캐싱 없음). FPS는 낮아도 무방.

사용법:
  python -m tools.verify_stability --frames 90
  python -m tools.verify_stability --frames 120 --label "bottle"
옵션:
  --frames  기록할 프레임 수(기본 90)
  --label   특정 라벨만 기록(미지정=전 객체)
  --out     프레임별 좌표 로그 CSV(기본 data/stability_log.csv)
"""

import argparse
import csv
import os

import numpy as np
import cv2

from vision.stream import WebcamStream, CAMERA_MATRIX
from vision.detector import ObjectDetector
from vision.segmentation.segmenter import ObjectSegmenter, SceneDepthAttacher
from vision.depth.depth_estimator import DepthEstimator
from vision.spatial.transformer import Spatial3DConverter
from vision.spatial.stabilizer import CoordinateStabilizer


def _pstd(a):
    if len(a) < 2:
        return 0.0
    m = sum(a) / len(a)
    return (sum((v - m) ** 2 for v in a) / len(a)) ** 0.5


def _mean_abs_step(a):
    if len(a) < 2:
        return 0.0
    return sum(abs(a[i + 1] - a[i]) for i in range(len(a) - 1)) / (len(a) - 1)


def main():
    ap = argparse.ArgumentParser(description="라이브 좌표 안정성 검증 (Task 4 / §7)")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="data/stability_log.csv")
    args = ap.parse_args()

    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth_engine = DepthEstimator()
    attacher = SceneDepthAttacher()
    spatial = Spatial3DConverter(CAMERA_MATRIX)
    stabilizer = CoordinateStabilizer()
    stream = WebcamStream()

    # label -> {"raw": {"x":[],"y":[],"z":[]}, "stab": {...}}
    series = {}
    rows = []
    fc = 0
    print(f"[안정성 검증] 정지 객체를 두고 {args.frames}프레임 기록합니다. 'q' 조기종료.")

    while fc < args.frames:
        ret, frame = stream.get_frame()
        if not ret:
            break
        fc += 1

        depth = depth_engine.get_depth_map(frame)
        yolo = detector.detect(frame)
        scene = detector.build_scene(yolo, frame, fc)
        ts = scene.get("frame_metadata", {}).get("timestamp", 0.0)
        _, scene, masks = segmenter.segment_objects(frame, scene)
        scene = attacher.attach_depth(scene, masks, depth)
        scene = spatial.process_scene_3d(scene)
        scene = stabilizer.process_scene(scene)   # raw_xyz + stabilized x/y/z 부여

        # 라벨별로 최대 마스크 객체 1개만 채택(중복 방지)
        best_by_label = {}
        for i, obj in enumerate(scene["objects"]):
            if obj is None:
                continue
            label = obj.get("label")
            if args.label is not None and label != args.label:
                continue
            sp = obj.get("spatial_3d")
            if not sp or "raw_xyz" not in sp:
                continue
            area = obj.get("sam", {}).get("mask_area", 0)
            if label not in best_by_label or area > best_by_label[label][0]:
                best_by_label[label] = (area, sp)

        for label, (_, sp) in best_by_label.items():
            rx, ry, rz = sp["raw_xyz"]
            sx, sy, sz = sp["x"], sp["y"], sp["z"]
            s = series.setdefault(label, {"raw": {"x": [], "y": [], "z": []},
                                          "stab": {"x": [], "y": [], "z": []}})
            s["raw"]["x"].append(rx); s["raw"]["y"].append(ry); s["raw"]["z"].append(rz)
            s["stab"]["x"].append(sx); s["stab"]["y"].append(sy); s["stab"]["z"].append(sz)
            rows.append([fc, round(ts, 4), label, rx, ry, rz, sx, sy, sz])

        # 표시
        disp = frame.copy()
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 40), (0, 0, 0), -1)
        labs = ", ".join(best_by_label.keys()) or "-"
        cv2.putText(disp, f"[{fc}/{args.frames}] tracking: {labs}", (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("CV_AR - Stability Verify", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()

    if not series:
        print("[경고] 기록된 객체 없음. 객체가 화면에 잡혔는지 확인.")
        return

    # 로그 저장
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp", "label", "raw_x", "raw_y", "raw_z", "stab_x", "stab_y", "stab_z"])
        w.writerows(rows)

    # 요약 리포트
    print("\n================ 좌표 안정성 요약 (단위 mm) ================")
    print(f"{'label':12} {'n':>4} {'mean_z(m)':>9} | {'stdZ_raw':>8} {'stdZ_stab':>9} | "
          f"{'|dZ|raw':>8} {'|dZ|stab':>8} | 감소")
    for label, s in series.items():
        n = len(s["stab"]["z"])
        mean_z = sum(s["stab"]["z"]) / n
        std_raw = _pstd(s["raw"]["z"]) * 1000
        std_stab = _pstd(s["stab"]["z"]) * 1000
        dz_raw = _mean_abs_step(s["raw"]["z"]) * 1000
        dz_stab = _mean_abs_step(s["stab"]["z"]) * 1000
        red = (1 - std_stab / std_raw) * 100 if std_raw > 1e-9 else 0.0
        print(f"{label:12} {n:>4} {mean_z:>9.3f} | {std_raw:>8.1f} {std_stab:>9.1f} | "
              f"{dz_raw:>8.1f} {dz_stab:>8.1f} | {red:>4.0f}%↓")
    print("===========================================================")
    print(f"프레임별 좌표 로그: {args.out}")
    print("해석: stdZ_stab(안정화 후 표준편차)이 작을수록 안정적. '감소'는 안정화가 지터를 줄인 비율.")


if __name__ == "__main__":
    main()
