"""정중앙 박스 투입 시나리오 기반 VLM 추론 결과 30개 수집 테스트.

전체 Vision/Geometry/VLM 파이프라인을 사용하되, 이 테스트 실행 중에만
SYSTEM_PROMPT를 확장하여 "웹캠 정중앙의 가상 박스에 상호작용 가능한 물건을 넣는"
목표를 VLM에게 부여한다.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test_center_box_affordance"
RESULT_PREFIX = "center_box_affordance_res"
TARGET_SAMPLES = 30


CENTER_BOX_SCENARIO_PROMPT = """

[테스트 전용 시나리오: 중앙 박스 투입 목표]
현재 웹캠 화면의 정중앙에는 가상의 고정 박스가 있다고 가정한다.
이미지에는 이 중앙 박스가 빨간 사각형으로 표시될 수 있다.
아바타의 목표는 안전하게 상호작용 가능한 물건을 선택하여 그 정중앙 박스 안에 넣는 것이다.
이 목표를 수행하기에 적절한 어포던스와 행동 정책을 판단하라.
객체가 중앙 박스보다 위쪽에 존재한다고 판단되면 `Reach up and take`를 affordances에 추가하라.
객체가 중앙 박스보다 아래쪽에 존재한다고 판단되면 `Bend down and pick up`을 affordances에 추가하라.
중앙 박스 자체는 실제 탐지 객체가 아니라 목표 위치이므로 JSON 결과 객체로 새로 만들지 마라.
"""


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def allocate_result_files() -> tuple[int, Path, Path]:
    """기존 결과를 덮어쓰지 않도록 다음 실험 번호의 txt/csv 경로를 만든다."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    max_run_id = 0
    for path in RESULT_DIR.glob(f"{RESULT_PREFIX}_*.csv"):
        suffix = path.stem.removeprefix(f"{RESULT_PREFIX}_")
        if suffix.isdigit():
            max_run_id = max(max_run_id, int(suffix))

    run_id = max_run_id + 1
    text_result = RESULT_DIR / f"{RESULT_PREFIX}_{run_id:03d}.txt"
    csv_result = RESULT_DIR / f"{RESULT_PREFIX}_{run_id:03d}.csv"
    return run_id, text_result, csv_result


def initialise_result_files(run_id: int, text_result: Path, csv_result: Path) -> None:
    started_at = now_text()
    text_result.write_text(
        "center_box_affordance: full pipeline VLM collection test\n"
        f"run_id: {run_id:03d}\n"
        "scenario: movable objects should be assigned affordances as if the avatar's goal is "
        "to place them into a virtual center box.\n"
        f"started_at: {started_at}\n"
        f"target_samples: {TARGET_SAMPLES}\n\n",
        encoding="utf-8",
    )
    with csv_result.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "run_id",
                "sample",
                "timestamp",
                "frame_count",
                "sam_frame_count",
                "object_count",
                "vlm_inference_seconds",
                "json",
            ]
        )


def append_result(
    run_id: int,
    text_result: Path,
    csv_result: Path,
    sample: int,
    frame_count: int,
    sam_frame_count: int,
    elapsed: float,
    result: dict,
) -> None:
    timestamp = now_text()
    compact_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
    object_count = len(result.get("results", []))

    with text_result.open("a", encoding="utf-8") as file:
        file.write(
            f"[run {run_id:03d} / sample {sample:02d}]\n"
            f"timestamp: {timestamp}\n"
            f"frame_count: {frame_count}\n"
            f"sam_frame_count: {sam_frame_count}\n"
            f"object_count: {object_count}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"json:\n{pretty_json}\n\n"
        )

    with csv_result.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                f"{run_id:03d}",
                sample,
                timestamp,
                frame_count,
                sam_frame_count,
                object_count,
                f"{elapsed:.6f}",
                compact_json,
            ]
        )


def center_box_bbox(frame) -> list[int]:
    """화면 정중앙 가상 박스의 픽셀 좌표 [x1, y1, x2, y2]를 반환한다."""
    height, width = frame.shape[:2]
    box_width = int(width * 0.25)
    box_height = int(height * 0.25)
    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    x2 = x1 + box_width
    y2 = y1 + box_height
    return [x1, y1, x2, y2]


def draw_center_box(frame):
    """VLM과 사용자 모두가 테스트 조건을 볼 수 있도록 중앙 목표 박스를 그린다."""
    import cv2

    display = frame.copy()
    x1, y1, x2, y2 = center_box_bbox(display)

    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        display,
        "CENTER BOX TARGET",
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    return display


def draw_object_boxes(frame, inputs):
    import cv2

    display = frame.copy()
    for inp in inputs:
        if not inp.bbox_2d:
            continue
        x1, y1, x2, y2 = map(int, inp.bbox_2d)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"Obj {inp.object_id}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return display


def main() -> int:
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    import cv2
    from PIL import Image

    import llm.interpreter as interpreter
    import llm.server_websocket as server
    from llm.feature_extractor import DEFAULT_CONTEXT, build_inputs_from_scene
    from llm.schemas import SemanticInterpretationBatchInput
    from vision.stream import WebcamStream

    run_id, text_result, csv_result = allocate_result_files()
    initialise_result_files(run_id, text_result, csv_result)

    original_system_prompt = interpreter.SYSTEM_PROMPT
    interpreter.SYSTEM_PROMPT = f"{original_system_prompt}\n{CENTER_BOX_SCENARIO_PROMPT}"

    server.init_vision_modules()
    sam_cache = {
        "last_labels": [],
        "last_bboxes": [],
        "last_masks_list": [],
        "last_sam_data": [],
        "cached_depth_map": None,
    }

    stream = WebcamStream()
    frame_count = 0
    sam_frame_count = 0
    sample = 0

    print("[center_box_affordance] 전체 파이프라인 + 중앙 박스 시나리오 테스트 시작")
    print(f"[center_box_affordance] run_id: {run_id:03d}")
    print(f"[center_box_affordance] TXT 저장: {text_result}")
    print(f"[center_box_affordance] CSV 저장: {csv_result}")
    print("[center_box_affordance] VLM 추론 결과 30개 수집 후 자동 종료합니다. q를 누르면 중단합니다.")

    try:
        while sample < TARGET_SAMPLES:
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % 5 != 0:
                preview = draw_center_box(frame)
                cv2.imshow("center box affordance test", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[center_box_affordance] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            sam_frame_count += 1
            scene_data = server.build_scene_graph_for_frame(
                frame,
                frame_count,
                sam_frame_count,
                sam_cache,
            )
            inputs = build_inputs_from_scene(scene_data)
            if not inputs:
                preview = draw_center_box(frame)
                cv2.imshow("center box affordance test", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[center_box_affordance] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            annotated_frame = draw_object_boxes(frame, inputs)
            annotated_frame = draw_center_box(annotated_frame)
            pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            context = (
                f"{DEFAULT_CONTEXT}. "
                "테스트 목표: 화면 정중앙의 가상 박스에 상호작용 가능한 물체를 넣기."
            )
            batch_input = SemanticInterpretationBatchInput(
                context=context,
                objects=inputs,
            )

            next_sample = sample + 1
            print(
                f"[center_box_affordance] sample {next_sample}/{TARGET_SAMPLES} "
                f"VLM 호출 (객체 {len(inputs)}개)"
            )
            started_at = time.perf_counter()
            batch_output = interpreter.interpret_batch(batch_input, image=pil_image)
            elapsed = time.perf_counter() - started_at

            result = batch_output.model_dump(mode="json")
            append_result(
                run_id,
                text_result,
                csv_result,
                next_sample,
                frame_count,
                sam_frame_count,
                elapsed,
                result,
            )
            sample = next_sample
            print(
                f"[center_box_affordance] sample {sample}/{TARGET_SAMPLES} 저장 완료 "
                f"({elapsed:.3f}초)"
            )

            cv2.imshow("center box affordance test", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[center_box_affordance] 사용자 입력 q로 중단합니다.")
                return 1
    except KeyboardInterrupt:
        print("\n[center_box_affordance] KeyboardInterrupt로 중단합니다.")
        return 1
    finally:
        interpreter.SYSTEM_PROMPT = original_system_prompt
        stream.release()
        cv2.destroyAllWindows()

    print("[center_box_affordance] 30개 VLM 추론 결과 수집 완료")
    print(f"[center_box_affordance] TXT: {text_result}")
    print(f"[center_box_affordance] CSV: {csv_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
