"""
공정 비교용 데이터 수집기 (본문 코드 미사용, 독립 실행).

웹캠에서 프레임을 캡처하고, 표준 YOLO(yolov8n.pt)로 person(agent) 박스와
target 객체 박스를 뽑아 manifest.csv에 기록한다. 이렇게 저장한 '같은 프레임'을
  (a) VLM-only 베이스라인(vlm_baseline.py)
  (b) 본문 Ours 파이프라인(run_ours.py)
양쪽에 동일하게 먹여 공정 비교한다.

■ 자동 모드(기본): 손으로 키를 누르지 않는다.
  실행 → 카운트다운(기본 5초) 동안 정해진 거리에 서 있으면,
  person+target이 둘 다 잡힐 때 --interval 간격으로 자동 연사한다.
  → 팔을 뻗어 키를 누를 필요가 없어 포즈/프레임이 오염되지 않는다.
  q 로 언제든 종료.

■ 수동 모드(--manual): 예전처럼 스페이스바로 한 장씩(디버깅용).

실행 예:
  python collect_frames.py --distance 0.30 --label "bottle" --shots 25
  python collect_frames.py --distance 0.80 --label "bottle" --shots 30 --countdown 8

manifest 컬럼: frame_id,image_path,label,agent_bbox,object_bbox,gt_distance_m,gt_executable
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from config import FRAMES_DIR, DIR, PPS_THRESHOLD_M

MANIFEST = DIR / "data" / "manifest.csv"
COLS = ["frame_id", "image_path", "label", "agent_bbox",
        "object_bbox", "gt_distance_m", "gt_executable"]


def _bbox_str(xyxy) -> str:
    return "|".join(str(int(v)) for v in xyxy)


def _append_manifest(row: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _next_id() -> int:
    if not MANIFEST.exists():
        return 0
    return sum(1 for _ in open(MANIFEST)) - 1  # 헤더 제외


def run(distance: float, label: str, shots: int, cam: int,
        auto: bool, interval: float, countdown: float) -> None:
    import cv2
    from ultralytics import YOLO

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(DIR.parent / "yolov8n.pt"))
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"카메라 {cam} 열기 실패")
        return
    # 서버(vision/stream.py)와 동일한 1280x720으로 고정 → 캘리브레이션 정합
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def capture(frame, person_box, target_box) -> str:
        fid = _next_id()
        fname = f"frame_{fid:05d}.jpg"
        cv2.imwrite(str(FRAMES_DIR / fname), frame)
        _append_manifest({
            "frame_id": fid,
            "image_path": f"data/frames/{fname}",
            "label": label,
            "agent_bbox": _bbox_str(person_box),
            "object_bbox": _bbox_str(target_box),
            "gt_distance_m": distance,
            "gt_executable": int(distance <= PPS_THRESHOLD_M),
        })
        return fname

    mode = "AUTO(무접촉)" if auto else "MANUAL(space)"
    print(f"[collect] label='{label}', GT거리={distance}m, 모드={mode}, 목표 {shots}장")
    if auto:
        print(f"  → {countdown:.0f}초 카운트다운 후, {interval}s 간격 자동 연사. "
              f"정해진 거리에 서 계세요. q=종료")
    else:
        print("  → 스페이스=촬영, q=종료")

    taken = 0
    start = time.time()
    last_shot = 0.0
    while taken < shots:
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, verbose=False)[0]
        names = res.names
        person_box = target_box = None
        p_conf = t_conf = -1.0
        for b in res.boxes:
            cls = names[int(b.cls)]
            conf = float(b.conf)
            xyxy = b.xyxy[0].tolist()
            if cls == "person" and conf > p_conf:
                person_box, p_conf = xyxy, conf
            elif cls == label and conf > t_conf:
                target_box, t_conf = xyxy, conf

        ready = bool(person_box and target_box)
        now = time.time()

        # ---- 자동/수동 촬영 판단 ----
        did_capture = False
        if auto:
            cd_left = countdown - (now - start)
            if cd_left > 0:
                banner = f"START IN {cd_left:0.1f}s  (자리로 이동)"
            elif not ready:
                banner = "WAIT: person+target 둘 다 잡히면 자동 촬영"
            elif (now - last_shot) >= interval:
                fname = capture(frame, person_box, target_box)
                taken += 1
                last_shot = now
                did_capture = True
                banner = f"SHOT {taken}/{shots}"
                print(f"  자동촬영 {taken}/{shots} → {fname}")
            else:
                wait = interval - (now - last_shot)
                banner = f"next in {wait:0.1f}s   {taken}/{shots}"
        else:
            banner = ("READY(space)" if ready
                      else f"need person={bool(person_box)} target={bool(target_box)}")

        disp = res.plot()
        color = (0, 255, 0) if (ready or not auto) else (0, 200, 255)
        cv2.putText(disp, f"{banner}  d={distance}m", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("collect_frames", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if not auto and key == ord(" ") and ready and not did_capture:
            fname = capture(frame, person_box, target_box)
            taken += 1
            print(f"  촬영 {taken}/{shots} → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"완료: {taken}장. manifest: {MANIFEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="공정 비교용 프레임+GT 수집")
    ap.add_argument("--distance", type=float, required=True, help="실측 GT 거리(m)")
    ap.add_argument("--label", required=True, help="타깃 객체 라벨 (예: 'bottle')")
    ap.add_argument("--shots", type=int, default=25)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--manual", action="store_true",
                    help="스페이스바 수동 촬영(디버깅용). 기본은 무접촉 자동 모드.")
    ap.add_argument("--interval", type=float, default=0.6,
                    help="자동 모드 촬영 간격(초)")
    ap.add_argument("--countdown", type=float, default=5.0,
                    help="자동 모드 시작 전 대기(초) — 자리로 이동할 시간")
    a = ap.parse_args()
    run(a.distance, a.label, a.shots, a.cam,
        auto=not a.manual, interval=a.interval, countdown=a.countdown)
