"""전체 Vision/Geometry/VLM 파이프라인의 Micro CoT 단어 제한 실험(30 -> 0)."""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test1"
RESULT_PREFIX = "test1_res"
WORD_LIMITS = tuple(range(30, -1, -1))


def reasoning_prompt(word_limit: int) -> str:
    if word_limit == 0:
        return """
[Micro CoT 출력 제한 - 실험 조건]
각 객체의 `reasoning`은 반드시 빈 문자열(`""`)로 출력하라.
`reasoning` 외에는 JSON 스키마가 요구하는 값만 출력하라.
"""
    return f"""
[Micro CoT 출력 제한 - 실험 조건]
내부 판단 과정이나 배경을 길게 설명하지 마라.
각 객체의 `reasoning`은 핵심 시각 근거와 결론만 담아 반드시 한국어 {word_limit}단어 이내로 작성하라.
단어 수는 공백으로 구분하여 계산한다.
`reasoning` 외에는 JSON 스키마가 요구하는 값만 출력하라.
"""


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
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    text_result.write_text(
        "test1: full Vision/Geometry/VLM pipeline reasoning word-limit experiment (30 -> 0)\n"
        f"run_id: {run_id:03d}\n"
        f"started_at: {started_at}\n\n",
        encoding="utf-8",
    )
    with csv_result.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            ["run_id", "experiment", "word_limit", "vlm_inference_seconds", "timestamp", "json"]
        )


def append_result(
    run_id: int,
    text_result: Path,
    csv_result: Path,
    experiment: int,
    word_limit: int,
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

    with text_result.open("a", encoding="utf-8") as file:
        file.write(
            f"[run {run_id:03d} / experiment {experiment:02d}]\n"
            f"word_limit: {word_limit}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"timestamp: {timestamp}\n"
            f"json:\n{pretty_json}\n\n"
        )

    with csv_result.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [f"{run_id:03d}", experiment, word_limit, f"{elapsed:.6f}", timestamp, compact_json]
        )


def main() -> int:
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    import cv2

    import llm.interpreter as interpreter
    import llm.server_websocket as server
    from vision.stream import WebcamStream

    run_id, text_result, csv_result = allocate_result_files()
    initialise_result_files(run_id, text_result, csv_result)
    experiment_done = threading.Event()
    result_lock = threading.Lock()
    next_result_index = 0
    original_interpret_batch = server.interpret_batch

    def measured_interpret_batch(batch_input, image=None):
        """전체 파이프라인 입력을 유지하고 VLM 호출 시간과 결과를 기록한다."""
        nonlocal next_result_index

        with result_lock:
            if next_result_index >= len(WORD_LIMITS):
                return original_interpret_batch(batch_input, image=image)
            word_limit = WORD_LIMITS[next_result_index]
            experiment = next_result_index + 1

        interpreter.USE_MICRO_COT = True
        interpreter.MICRO_COT_PROMPT = reasoning_prompt(word_limit)
        print(
            f"[test1] 실험 {experiment}/{len(WORD_LIMITS)} 시작: "
            f"전체 파이프라인 + reasoning 최대 {word_limit}단어"
        )

        started_at = time.perf_counter()
        output = original_interpret_batch(batch_input, image=image)
        elapsed = time.perf_counter() - started_at
        result = output.model_dump(mode="json")

        with result_lock:
            append_result(run_id, text_result, csv_result, experiment, word_limit, elapsed, result)
            next_result_index += 1
            print(
                f"[test1] 실험 {experiment}/{len(WORD_LIMITS)} "
                f"저장 완료 ({elapsed:.3f}초)"
            )
            if next_result_index == len(WORD_LIMITS):
                experiment_done.set()
        return output

    # server의 VLM worker가 동일한 전체 파이프라인 입력으로 측정 래퍼를 호출하게 한다.
    server.interpret_batch = measured_interpret_batch
    # 정적인 장면에서도 30→0의 모든 조건을 순서대로 실행한다.
    server.is_significant_change = lambda _previous, _current: True

    threading.Thread(target=server.ai_worker_thread, daemon=True).start()
    threading.Thread(target=server.vlm_worker_thread, daemon=True).start()

    stream = WebcamStream()
    print(
        "[test1] YOLO/SAM/Depth/Geometry/VLM 전체 파이프라인 실험 시작. "
        "q를 누르면 중단합니다."
    )
    print(f"[test1] run_id: {run_id:03d}")
    print(f"[test1] TXT 저장: {text_result}")
    print(f"[test1] CSV 저장: {csv_result}")
    try:
        while not experiment_done.is_set():
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
            cv2.imshow("test1 - full pipeline, reasoning word limit 30 to 0", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[test1] 사용자 요청으로 실험을 중단합니다.")
                return 1
    except KeyboardInterrupt:
        print("\n[test1] 사용자 요청으로 실험을 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print(f"[test1] 31개 실험 완료: {text_result} / {csv_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
