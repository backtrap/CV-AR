"""전체 파이프라인 실행 중 FAST_STREAM 데이터와 VLM SUCCESS 데이터를 분리 저장하는 임시 테스트."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test_temp"
FAST_STREAM_TEXT = RESULT_DIR / "fast_stream_data.txt"
VLM_RESULT_TEXT = RESULT_DIR / "vlm_result_data.txt"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def initialise_result_files() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = now_text()

    FAST_STREAM_TEXT.write_text(
        "test_temp: FAST_STREAM payload log\n"
        f"started_at: {started_at}\n\n",
        encoding="utf-8",
    )
    VLM_RESULT_TEXT.write_text(
        "test_temp: VLM SUCCESS payload log\n"
        f"started_at: {started_at}\n\n",
        encoding="utf-8",
    )


def append_json_block(path: Path, header: str, payload: object) -> None:
    pretty_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{header}\n{pretty_payload}\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Vision/Geometry/VLM pipeline and separately log "
            "FAST_STREAM and VLM SUCCESS WebSocket payloads."
        )
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="테스트 지속 시간(초). 0이면 q 입력 전까지 계속 실행합니다.",
    )
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    import cv2

    import llm.server_websocket as server
    from vision.stream import WebcamStream

    initialise_result_files()

    log_lock = threading.Lock()
    original_broadcast_message = server.broadcast_message

    def logging_broadcast_message(msg_dict: dict) -> None:
        """WebSocket으로 보내려는 메시지를 상태별 txt 파일에 저장한 뒤 원래대로 전송한다."""
        status = msg_dict.get("status")
        timestamp = now_text()

        with log_lock:
            if status == "FAST_STREAM":
                append_json_block(
                    FAST_STREAM_TEXT,
                    f"[{timestamp}] status=FAST_STREAM",
                    msg_dict.get("data"),
                )
            elif status == "SUCCESS":
                append_json_block(
                    VLM_RESULT_TEXT,
                    f"[{timestamp}] status=SUCCESS",
                    msg_dict.get("data"),
                )
            elif status == "API_LIMIT_EXCEEDED":
                append_json_block(
                    VLM_RESULT_TEXT,
                    f"[{timestamp}] status=API_LIMIT_EXCEEDED",
                    msg_dict.get("data"),
                )

        original_broadcast_message(msg_dict)

    server.broadcast_message = logging_broadcast_message

    threading.Thread(target=server.start_websocket_server, daemon=True).start()
    threading.Thread(target=server.ai_worker_thread, daemon=True).start()
    threading.Thread(target=server.vlm_worker_thread, daemon=True).start()

    stream = WebcamStream()
    started_at = time.perf_counter()
    print("[test_temp] 전체 Vision/Geometry/VLM 파이프라인 테스트 시작")
    print(f"[test_temp] FAST_STREAM 저장: {FAST_STREAM_TEXT}")
    print(f"[test_temp] VLM SUCCESS 저장: {VLM_RESULT_TEXT}")
    print("[test_temp] q를 누르면 종료합니다.")

    try:
        while True:
            if args.duration > 0 and time.perf_counter() - started_at >= args.duration:
                print(f"[test_temp] 지정한 {args.duration:.1f}초가 지나 종료합니다.")
                break

            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            with server.frame_lock:
                server.latest_frame = frame

            with server.annotated_lock:
                display = (
                    server.annotated_frame_to_display
                    if server.annotated_frame_to_display is not None
                    else frame
                )

            cv2.imshow("test_temp - full pipeline logger", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[test_temp] 사용자 입력 q로 종료합니다.")
                break
    except KeyboardInterrupt:
        print("\n[test_temp] KeyboardInterrupt로 종료합니다.")
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print("[test_temp] 저장 완료")
    print(f"[test_temp] FAST_STREAM: {FAST_STREAM_TEXT}")
    print(f"[test_temp] VLM SUCCESS: {VLM_RESULT_TEXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
