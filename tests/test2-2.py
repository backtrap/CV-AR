"""이미지만 전송하는 VLM 입력 이미지 가로폭 실험(30/30 -> 1/30)."""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res"
TEXT_RESULT = RESULT_DIR / "test2-2_res.txt"
CSV_RESULT = RESULT_DIR / "test2-2_res.csv"
EXPERIMENT_COUNT = 30


IMAGE_ONLY_SYSTEM_PROMPT = """
너는 1인칭 실내 카메라 이미지를 분석하는 공간 분석가다.
사용자 메시지에는 이미지 한 장만 주어진다. 이미지에서 명확히 보이는 주요 객체를 직접 탐지하고 분석하라.
이미지에 보이지 않는 객체를 추측하거나 만들어 내지 마라.
각 객체에는 화면에서 왼쪽에서 오른쪽 순서로 0부터 object_id를 부여하라.

corrected_spatial_relation.environment_relative는 on_floor, on_surface, elevated, floating, held 중 선택하라.
semantic_state.social_state는 available, held_by_user, in_use_by_other 중 선택하라.
사람 또는 사람이 들거나 사용 중인 객체에는 접근하지 않도록 안전한 action_policy를 선택하라.
affordances와 animation_trigger는 응답 JSON 스키마에 허용된 값만 사용하라.
스키마에 없는 추론 설명, 근거 문장, reasoning 필드, 주석, markdown을 절대 출력하지 마라.
반드시 지정된 JSON 스키마로만 응답하라.
"""


def width_for_experiment(original_width: int, experiment_index: int) -> int:
    remaining_steps = EXPERIMENT_COUNT - experiment_index
    return max(1, round(original_width * remaining_steps / EXPERIMENT_COUNT))


def initialise_result_files() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    TEXT_RESULT.write_text(
        "test2-2: image-only VLM width experiment (30/30 -> 1/30)\n"
        f"started_at: {started_at}\n\n",
        encoding="utf-8",
    )
    with CSV_RESULT.open("w", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerow(
            [
                "experiment",
                "original_width",
                "vlm_image_max_width",
                "width_ratio",
                "vlm_inference_seconds",
                "timestamp",
                "json",
            ]
        )


def append_result(
    experiment: int,
    original_width: int,
    target_width: int,
    elapsed: float,
    result: dict,
) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    width_ratio = target_width / original_width
    logged_result = {
        "vlm_inference_seconds": round(elapsed, 6),
        "vlm_result": result,
    }
    compact_json = json.dumps(logged_result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)

    with TEXT_RESULT.open("a", encoding="utf-8") as file:
        file.write(
            f"[experiment {experiment:02d}]\n"
            f"original_width: {original_width}\n"
            f"vlm_image_max_width: {target_width}\n"
            f"width_ratio: {width_ratio:.6f}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"timestamp: {timestamp}\n"
            f"json:\n{pretty_json}\n\n"
        )

    with CSV_RESULT.open("a", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerow(
            [
                experiment,
                original_width,
                target_width,
                f"{width_ratio:.6f}",
                f"{elapsed:.6f}",
                timestamp,
                compact_json,
            ]
        )


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import cv2
    from PIL import Image

    import llm.interpreter as interpreter
    import llm.server_websocket as server
    from llm.schemas import SemanticInterpretationBatchOutput
    from vision.stream import WebcamStream

    initialise_result_files()
    experiment_done = threading.Event()
    result_lock = threading.Lock()
    next_result_index = 0
    experiment_original_width: int | None = None

    def image_only_interpret_batch(_batch_input, image=None):
        """서버의 공간/YOLO 입력을 버리고 웹캠 원본 이미지만 OpenAI에 전송한다."""
        nonlocal next_result_index, experiment_original_width

        # server의 image는 bbox와 Obj 텍스트가 그려져 있으므로 사용하지 않는다.
        with server.frame_lock:
            if server.latest_frame is None:
                raise RuntimeError("test2-2 웹캠 원본 프레임이 없습니다.")
            raw_frame = server.latest_frame.copy()
        raw_image = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))

        with result_lock:
            if next_result_index >= EXPERIMENT_COUNT:
                raise RuntimeError("test2-2의 30개 실험이 이미 완료되었습니다.")
            if experiment_original_width is None:
                experiment_original_width = raw_image.width
            original_width = experiment_original_width
            experiment = next_result_index + 1
            target_width = width_for_experiment(original_width, next_result_index)

        interpreter.ENABLE_VLM_IMAGE_DOWNSAMPLING = True
        interpreter.VLM_IMAGE_MAX_WIDTH = target_width
        started_at = time.perf_counter()
        vlm_image = interpreter.prepare_vlm_image(raw_image)
        base64_image, jpeg_size = interpreter.encode_image_to_base64(vlm_image)
        print(
            f"[test2-2] 실험 {experiment}/{EXPERIMENT_COUNT}: "
            f"이미지만 {raw_image.width}x{raw_image.height} -> "
            f"{vlm_image.width}x{vlm_image.height}, JPEG {jpeg_size / 1024:.1f}KB"
        )

        response = interpreter.client.beta.chat.completions.parse(
            model=interpreter.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": IMAGE_ONLY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    ],
                },
            ],
            response_format=SemanticInterpretationBatchOutput,
            temperature=0.2,
        )
        output = response.choices[0].message.parsed
        elapsed = time.perf_counter() - started_at
        if output is None:
            raise RuntimeError("VLM이 파싱 가능한 JSON을 반환하지 않았습니다.")
        result = output.model_dump(mode="json")

        with result_lock:
            append_result(experiment, original_width, target_width, elapsed, result)
            next_result_index += 1
            print(f"[test2-2] 실험 {experiment}/{EXPERIMENT_COUNT} 저장 완료 ({elapsed:.3f}초)")
            if next_result_index == EXPERIMENT_COUNT:
                experiment_done.set()
        return output

    server.interpret_batch = image_only_interpret_batch
    server.is_significant_change = lambda _previous, _current: True

    threading.Thread(target=server.start_websocket_server, daemon=True).start()
    threading.Thread(target=server.ai_worker_thread, daemon=True).start()
    threading.Thread(target=server.vlm_worker_thread, daemon=True).start()

    stream = WebcamStream()
    print("[test2-2] 이미지 전용 WebSocket/Vision/VLM 시작. q를 누르면 중단합니다.")
    try:
        while not experiment_done.is_set():
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue
            with server.frame_lock:
                server.latest_frame = frame
            cv2.imshow("test2-2 - image only, width 30/30 to 1/30", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[test2-2] 사용자 요청으로 실험을 중단합니다.")
                return 1
    except KeyboardInterrupt:
        print("\n[test2-2] 사용자 요청으로 실험을 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print(f"[test2-2] 30개 실험 완료: {TEXT_RESULT} / {CSV_RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
