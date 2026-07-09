"""
[Task 6] Depth 스케일 실측 도구 — 알려진 거리에서 '모델 원시 예측 깊이'(pred, 보정 전)를 수집.

캘리브레이션은 보정 전(pred) 값이 필요하므로, 이 스크립트는 depth 보정 계수를
런타임에서 강제로 scale=1.0, offset=0.0 으로 되돌려 순수 pred를 로깅한다.
(depth_scale.json 파일은 건드리지 않는다.)

사용법 (거리마다 한 번씩 실행, 같은 CSV에 누적):
  python -m tools.measure_depth --true 0.5
  python -m tools.measure_depth --true 1.0
  python -m tools.measure_depth --true 1.5   ...
  # 각 실행: 화면에서 대상 객체를 조준 → 자동으로 N프레임 수집 후 1행 추가
그다음:
  python -m tools.calibrate_depth_scale data/depth_calib.csv --write

옵션:
  --true   (필수) 줄자로 잰 카메라-객체 실제 거리(m)
  --label  대상 클래스명(예: cell phone). 미지정 시 '가장 큰 마스크' 객체 사용
  --frames 수집 프레임 수(기본 60)
  --out    출력 CSV(기본 data/depth_calib.csv)
"""

import argparse
import csv
import os

import numpy as np
import cv2

from vision.stream import WebcamStream
from vision.detector import ObjectDetector
from vision.segmentation.segmenter import ObjectSegmenter
from vision.depth.depth_estimator import DepthEstimator, robust_representative_depth


def main():
    ap = argparse.ArgumentParser(description="알려진 거리에서 원시 depth pred 수집 (Task 6)")
    ap.add_argument("--true", type=float, required=True, dest="true_m", help="실제 거리(m)")
    ap.add_argument("--label", default=None, help="대상 클래스명(미지정=최대 마스크)")
    ap.add_argument("--frames", type=int, default=60, help="수집 프레임 수")
    ap.add_argument("--out", default="data/depth_calib.csv", help="출력 CSV")
    args = ap.parse_args()

    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    estimator = DepthEstimator()
    # 핵심: 순수 pred 확보를 위해 보정 계수를 항등으로 되돌린다(파일은 그대로).
    estimator.depth_scale, estimator.depth_offset = 1.0, 0.0
    print(f"[측정] 보정 OFF (scale=1, offset=0). 실제거리 {args.true_m}m 조준 후 수집합니다.")

    stream = WebcamStream()
    preds = []
    frame_count = 0
    print("-> 대상이 화면 중앙에 안정적으로 잡히면 자동 수집됩니다. 'q'로 조기 종료.")

    while len(preds) < args.frames:
        ret, frame = stream.get_frame()
        if not ret:
            break
        frame_count += 1

        depth_data = estimator.get_depth_map(frame)   # 이제 pred 원시값
        yolo_result = detector.detect(frame)
        scene = detector.build_scene(yolo_result, frame, frame_count)
        # 대상 라벨만 남기고 나머지는 제외 → SAM/표시/측정 모두 대상만 처리
        if args.label is not None:
            scene["objects"] = [o for o in scene["objects"] if o["label"] == args.label]
        if not scene["objects"]:
            note = f"'{args.label}' 미검출" if args.label else "감지된 객체 없음"
            _annotate(frame, scene, None, None, len(preds), args.frames, note)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        _, scene, masks = segmenter.segment_objects(frame, scene)
        if not masks:
            _annotate(frame, scene, None, None, len(preds), args.frames, "마스크 없음")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # 대상 선택: 라벨 지정 시 그 라벨 중 최대 마스크, 아니면 전체 최대 마스크
        idx = _pick_target(scene, masks, args.label)
        if idx is None:
            _annotate(frame, scene, masks, None, len(preds), args.frames,
                      f"'{args.label}' 미검출")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        raw_depth = depth_data["raw_depth"]
        m = masks[idx]
        if raw_depth.shape != m.shape:
            m = cv2.resize(m.astype(np.uint8), (raw_depth.shape[1], raw_depth.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        rep = robust_representative_depth(raw_depth[m])["representative_depth"]
        if rep > 0:
            preds.append(rep)

        _annotate(frame, scene, masks, idx, len(preds), args.frames, f"pred={rep:.3f}m")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()

    if not preds:
        print("[경고] 수집된 pred 없음. 대상/조명 확인 후 재시도.")
        return

    pred_median = float(np.median(preds))
    pred_std = float(np.std(preds))
    print(f"\n실제 {args.true_m}m | 수집 {len(preds)}프레임 | pred median={pred_median:.4f}m std={pred_std*1000:.1f}mm")

    header = ["true_distance_m", "pred_depth_m", "n_frames", "pred_std_m"]
    new_row = [args.true_m, round(pred_median, 4), len(preds), round(pred_std, 4)]

    # upsert: 같은 거리(true_distance_m) 행이 이미 있으면 덮어쓴다(재측정 시 중복 방지).
    rows, replaced = [], False
    if os.path.exists(args.out):
        with open(args.out, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        for r in reader[1:] if reader else []:
            if not r:
                continue
            if abs(float(r[0]) - args.true_m) < 1e-6:
                replaced = True
                continue  # 기존 동일 거리 행 제거
            rows.append(r)
    rows.append([str(x) for x in new_row])
    rows.sort(key=lambda r: float(r[0]))  # 거리순 정렬

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    action = "덮어씀(재측정)" if replaced else "추가"
    print(f"✅ {args.out} 에 {args.true_m}m 1행 {action}. (총 {len(rows)}개 거리)")


def _pick_target(scene, masks, label):
    best, best_area = None, -1
    for i, obj in enumerate(scene["objects"]):
        if i >= len(masks):
            break
        if label is not None and obj["label"] != label:
            continue
        area = int(np.sum(masks[i]))
        if area > best_area:
            best, best_area = i, area
    return best


def _annotate(frame, scene, masks, target_idx, got, total, note):
    """탐지 상태를 눈으로 확인할 수 있게 bbox/마스크/라벨을 화면에 그린다."""
    disp = frame.copy()
    objs = scene.get("objects", []) if scene else []

    # 대상 마스크를 초록으로 오버레이
    if masks is not None and target_idx is not None and target_idx < len(masks):
        m = masks[target_idx]
        if m.shape[:2] == disp.shape[:2]:
            overlay = disp.copy()
            overlay[m] = (0, 200, 0)
            disp = cv2.addWeighted(disp, 0.6, overlay, 0.4, 0)

    # 모든 YOLO 박스: 대상은 굵은 초록, 나머지는 얇은 회색
    for i, obj in enumerate(objs):
        bb = obj.get("yolo", {}).get("bbox_2d")
        if not bb:
            continue
        x1, y1, x2, y2 = map(int, bb)
        is_target = (i == target_idx)
        color = (0, 255, 0) if is_target else (180, 180, 180)
        thick = 3 if is_target else 1
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, thick)
        conf = obj.get("yolo", {}).get("confidence", 0)
        cv2.putText(disp, f"{obj.get('label','?')} {conf:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 상단 상태바
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(disp, f"[{got}/{total}]  {note}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    labels = ", ".join(sorted({o.get("label", "?") for o in objs})) or "-"
    cv2.putText(disp, f"detected: {labels}", (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("CV_AR - Depth Calibration Capture", disp)


if __name__ == "__main__":
    main()
