"""이미지만 사용하는 VLM의 JPEG 압축 품질 실험(100 -> 13)."""

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
TEXT_RESULT = RESULT_DIR / "test2-1_res.txt"
CSV_RESULT = RESULT_DIR / "test2-1_res.csv"
JPEG_QUALITIES = [
    100, 97, 94, 91, 88,
    85, 82, 79, 76, 73,
    70, 67, 64, 61, 58,
    55, 52, 49, 46, 43,
    40, 37, 34, 31, 28,
    25, 22, 19, 16, 13,
]
EXPERIMENT_COUNT = len(JPEG_QUALITIES)


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


def initialise_result_files() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    TEXT_RESULT.write_text(
        "test2-1: image-only VLM JPEG quality experiment (100 -> 13)\n"
        f"started_at: {started_at}\n\n",
        encoding="utf-8",
    )
    with CSV_RESULT.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "experiment",
                "jpeg_quality",
                "jpeg_size_bytes",
                "vlm_inference_seconds",
                "timestamp",
                "json",
            ]
        )


def append_result(
    experiment: int,
    jpeg_quality: int,
    jpeg_size_bytes: int,
    elapsed: float,
    result: dict,
) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    logged_result = {
        "vlm_inference_seconds": round(elapsed, 6),
        "vlm_result": result,
    }
    compact_json = json.dumps(logged_result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)

    with TEXT_RESULT.open("a", encoding="utf-8") as file:
        file.write(
            f"[experiment {experiment:02d}]\n"
            f"jpeg_quality: {jpeg_quality}\n"
            f"jpeg_size_bytes: {jpeg_size_bytes}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"timestamp: {timestamp}\n"
            f"json:\n{pretty_json}\n\n"
        )

    with CSV_RESULT.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                experiment,
                jpeg_quality,
                jpeg_size_bytes,
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

    def image_only_interpret_batch(_batch_input, image=None):
        """Vision/Geometry 입력을 버리고 웹캠 원본 이미지만 OpenAI에 전송한다."""
        nonlocal next_result_index

        # server의 image에는 bbox와 Obj 텍스트가 있으므로 원본 프레임을 다시 가져온다.
        with server.frame_lock:
            if server.latest_frame is None:
                raise RuntimeError("test2-1 웹캠 원본 프레임이 없습니다.")
            raw_frame = server.latest_frame.copy()
        raw_image = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))

        with result_lock:
            if next_result_index >= EXPERIMENT_COUNT:
                raise RuntimeError("test2-1의 30개 실험이 이미 완료되었습니다.")
            experiment = next_result_index + 1
            jpeg_quality = JPEG_QUALITIES[next_result_index]

        # 해상도는 원본 그대로 유지하고 JPEG 품질만 변경한다.
        interpreter.ENABLE_VLM_IMAGE_DOWNSAMPLING = True
        interpreter.VLM_IMAGE_MAX_WIDTH = raw_image.width
        interpreter.VLM_JPEG_QUALITY = jpeg_quality

        started_at = time.perf_counter()
        base64_image, jpeg_size_bytes = interpreter.encode_image_to_base64(raw_image)
        print(
            f"[test2-1] 실험 {experiment}/{EXPERIMENT_COUNT}: "
            f"JPEG 품질 {jpeg_quality}, {jpeg_size_bytes / 1024:.1f}KB"
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
            append_result(
                experiment,
                jpeg_quality,
                jpeg_size_bytes,
                elapsed,
                result,
            )
            next_result_index += 1
            print(f"[test2-1] 실험 {experiment}/{EXPERIMENT_COUNT} 저장 완료 ({elapsed:.3f}초)")
            if next_result_index == EXPERIMENT_COUNT:
                experiment_done.set()
        return output

    server.interpret_batch = image_only_interpret_batch
    server.is_significant_change = lambda _previous, _current: True

    threading.Thread(target=server.start_websocket_server, daemon=True).start()
    threading.Thread(target=server.ai_worker_thread, daemon=True).start()
    threading.Thread(target=server.vlm_worker_thread, daemon=True).start()

    stream = WebcamStream()
    print("[test2-1] 이미지 전용 JPEG 품질 실험 시작. q를 누르면 중단합니다.")
    try:
        while not experiment_done.is_set():
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue
            with server.frame_lock:
                server.latest_frame = frame
            cv2.imshow("test2-1 - image only, JPEG quality 100 to 13", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[test2-1] 사용자 요청으로 실험을 중단합니다.")
                return 1
    except KeyboardInterrupt:
        print("\n[test2-1] 사용자 요청으로 실험을 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print(f"[test2-1] 30개 실험 완료: {TEXT_RESULT} / {CSV_RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
