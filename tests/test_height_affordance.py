"""높이 차이에 따른 어포던스 부여 VLM 테스트.

높은 위치의 물체와 낮은 위치의 물체가 동시에 보이는 장면에서
VLM이 `Reach up and take`, `Bend down and pick up`을 적절히 affordances에
포함하는지 확인하기 위한 전체 파이프라인 테스트.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test_height_affordance"
RESULT_PREFIX = "height_affordance_res"
TARGET_SAMPLES = 30


HEIGHT_AFFORDANCE_SCENARIO_PROMPT = """

[테스트 전용 시나리오: 높이 차이에 따른 수거 행동 판단]
아바타는 실내 공간에 흩어진 상호작용 가능한 물체들을 정리 위치로 옮기는 상황이다.
이 테스트의 목적은 높은 위치의 물체와 낮은 위치의 물체가 동시에 보일 때,
VLM이 3D 좌표와 이미지상 높이 차이를 근거로 적절한 affordance를 부여하는지 확인하는 것이다.

각 객체에 대해 다음 기준을 반영하라.
1. 물체가 높은 선반, 높은 상자, 책상 위쪽, 화면 상단, 또는 사람/아바타 기준 높은 위치에 있다고 판단되면
   `Reach up and take`를 affordances에 포함하라.
2. 물체가 바닥, 낮은 선반, 의자 아래쪽, 화면 하단, 또는 사람/아바타 기준 낮은 위치에 있다고 판단되면
   `Bend down and pick up`을 affordances에 포함하라.
3. 물체의 형태가 병, 컵, 텀블러처럼 원통형이면 높이 기반 affordance와 별개로
   `Cylindrical grasp to move`도 함께 포함할 수 있다.
4. 작고 얇거나 손가락으로 집기 적합한 물체는 `Pinch grasp to move`도 함께 포함할 수 있다.
5. affordances는 하나만 고르는 값이 아니므로, 현재 객체에 가능한 행동을 모두 포함하라.
6. `animation_trigger`는 affordances 중 지금 먼저 수행해야 할 단일 행동이다.
   높은 위치 물체는 `Reach up and take`, 낮은 위치 물체는 `Bend down and pick up`을 우선 선택하라.
7. 사람, 손에 들린 물체, 타인이 사용 중인 물체는 수거 대상이 아니므로 접근하지 마라.
8. 중요: 입력 objects 배열에 포함된 모든 객체에 대해 반드시 하나의 결과를 반환하라.
   높이 판단 대상이 아니거나 정리 대상이 아닌 객체도 절대 생략하지 마라.
   그런 객체는 `action_policy=IGNORE`, `animation_trigger=None`으로 출력하라.
   `results` 배열 길이는 입력 objects 배열 길이와 정확히 같아야 한다.
"""


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def allocate_result_files() -> tuple[int, Path, Path]:
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
        "height_affordance: full pipeline VLM collection test\n"
        f"run_id: {run_id:03d}\n"
        "scenario: assign height-aware affordances for high and low objects in the same scene.\n"
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


def draw_guides(frame):
    """상/중/하 높이 영역을 VLM과 사용자가 볼 수 있게 안내선으로 표시한다."""
    import cv2

    display = frame.copy()
    height, width = display.shape[:2]
    upper_y = int(height * 0.33)
    lower_y = int(height * 0.66)

    cv2.line(display, (0, upper_y), (width, upper_y), (255, 0, 0), 2)
    cv2.line(display, (0, lower_y), (width, lower_y), (0, 0, 255), 2)
    cv2.putText(
        display,
        "HIGH AREA: Reach up and take",
        (15, max(24, upper_y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        display,
        "LOW AREA: Bend down and pick up",
        (15, min(height - 14, lower_y + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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
    interpreter.SYSTEM_PROMPT = f"{original_system_prompt}\n{HEIGHT_AFFORDANCE_SCENARIO_PROMPT}"

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

    print("[height_affordance] 전체 파이프라인 + 높이 어포던스 테스트 시작")
    print(f"[height_affordance] run_id: {run_id:03d}")
    print(f"[height_affordance] TXT 저장: {text_result}")
    print(f"[height_affordance] CSV 저장: {csv_result}")
    print("[height_affordance] 높은 물체와 낮은 물체를 동시에 배치하세요. 30개 수집 후 종료합니다.")

    try:
        while sample < TARGET_SAMPLES:
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % 5 != 0:
                preview = draw_guides(frame)
                cv2.imshow("height affordance test", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[height_affordance] 사용자 입력 q로 중단합니다.")
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
                preview = draw_guides(frame)
                cv2.imshow("height affordance test", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[height_affordance] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            annotated_frame = draw_object_boxes(frame, inputs)
            annotated_frame = draw_guides(annotated_frame)
            pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            context = (
                f"{DEFAULT_CONTEXT}. "
                "테스트 목표: 높은 위치의 물체와 낮은 위치의 물체를 동시에 보고, "
                "높이에 따른 수거/정리 어포던스를 부여하기."
            )
            batch_input = SemanticInterpretationBatchInput(
                context=context,
                objects=inputs,
            )

            next_sample = sample + 1
            print(
                f"[height_affordance] sample {next_sample}/{TARGET_SAMPLES} "
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
                f"[height_affordance] sample {sample}/{TARGET_SAMPLES} 저장 완료 "
                f"({elapsed:.3f}초)"
            )

            cv2.imshow("height affordance test", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[height_affordance] 사용자 입력 q로 중단합니다.")
                return 1
    except KeyboardInterrupt:
        print("\n[height_affordance] KeyboardInterrupt로 중단합니다.")
        return 1
    finally:
        interpreter.SYSTEM_PROMPT = original_system_prompt
        stream.release()
        cv2.destroyAllWindows()

    print("[height_affordance] 30개 VLM 추론 결과 수집 완료")
    print(f"[height_affordance] TXT: {text_result}")
    print(f"[height_affordance] CSV: {csv_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
