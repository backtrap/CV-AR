"""
VLM-only 베이스라인 (SpatialVLM류 / GPT-4V 직접 거리·결정 판단).

동일 프레임과 동일 YOLO 박스를 입력으로 받아, 거리와 실행가능성을
VLM에게 '직접' 물어본다. 여기에는 어떤 기하(Depth/3D) 계산도 개입하지 않는다.
→ 본문 Ours(Depth+PPS)와의 유일한 차이 = "거리·결정의 출처".

입력: manifest CSV
  columns: frame_id,image_path,label,agent_bbox,object_bbox,gt_distance_m,gt_executable
  bbox 형식: "x1|y1|x2|y2" (픽셀)

출력: results/vlm_results.csv
  frame_id,label,vlm_distance_m,vlm_executable,flip_rate,latency_ms,raw_distances,raw_execs

재현성을 위해 temperature=0, 모델 버전 고정(config.VLM_MODEL).
같은 프레임을 N_REPEAT회 질의해 결정 안정성(flip-rate)을 측정한다.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

from config import OPENAI_API_KEY, VLM_MODEL, N_REPEAT, PPS_THRESHOLD_M, DIR, RESULTS_DIR

# 거리 회귀 프롬프트 (SpatialVLM 스타일: 정량적 거리 질의)
DISTANCE_PROMPT = (
    "You are given an image. A '{agent}' (the agent/user) is at pixel box {abox} "
    "and a '{label}' object is at pixel box {obox} (format x1,y1,x2,y2).\n"
    "Estimate the straight-line physical distance between the PERSON (agent) and the "
    "'{label}', in METERS. Consider real-world object sizes for scale.\n"
    "Answer with ONLY a single number (e.g. 0.65). No words, no units."
)

# 실행가능성 결정 프롬프트 (peripersonal space 판단)
DECISION_PROMPT = (
    "You are given an image. A '{agent}' (the agent/user) is at pixel box {abox} "
    "and a '{label}' object is at pixel box {obox} (format x1,y1,x2,y2).\n"
    "Can the agent physically reach and interact with the '{label}' RIGHT NOW, "
    "i.e. is it within arm's reach / peripersonal space (about {thr} m)?\n"
    'Answer strictly as JSON: {{"executable": true or false}}.'
)


def _client():
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 미설정 (.env 확인)")
    # max_retries: 429(rate limit) 발생 시 SDK가 Retry-After를 존중해 자동 백오프
    return OpenAI(api_key=OPENAI_API_KEY, max_retries=10)


def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _ask(client, image_b64: str, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
    )
    latency = (time.perf_counter() - t0) * 1000.0
    return resp.choices[0].message.content or "", latency


def parse_distance(text: str) -> float | None:
    m = re.search(r"-?\d+\.?\d*", text.replace(",", "."))
    return float(m.group()) if m else None


def parse_decision(text: str) -> int | None:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return int(bool(json.loads(m.group()).get("executable")))
    except Exception:
        pass
    low = text.lower()
    if "true" in low:
        return 1
    if "false" in low:
        return 0
    return None


FIELDS = ["frame_id", "label", "vlm_distance_m", "vlm_executable", "flip_rate",
          "latency_ms", "raw_distances", "raw_execs", "raw_dist_text"]


def run(manifest_path: Path, out_path: Path) -> None:
    import time
    import numpy as np

    client = _client()
    throttle = float(os.environ.get("BASELINE_THROTTLE", "2.0"))  # TPM 보호용 프레임간 대기(초)
    rows = list(csv.DictReader(open(manifest_path, newline="")))
    print(f"[vlm_baseline] {len(rows)}개 항목, 모델={VLM_MODEL}, N_REPEAT={N_REPEAT}, "
          f"throttle={throttle}s")

    # 증분 저장: 크래시가 나도 진행분은 보존
    f = open(out_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    f.flush()

    n_written = 0
    for i, row in enumerate(rows):
        img_path = (DIR / row["image_path"]) if not Path(row["image_path"]).is_absolute() \
            else Path(row["image_path"])
        if not img_path.exists():
            print(f"  [skip] 이미지 없음: {img_path}")
            continue
        b64 = encode_image(img_path)
        fmt = dict(agent="person", label=row["label"],
                   abox=row["agent_bbox"].replace("|", ","),
                   obox=row["object_bbox"].replace("|", ","),
                   thr=PPS_THRESHOLD_M)

        dists, execs, lat_total = [], [], 0.0
        first_dtxt = ""
        for k in range(N_REPEAT):
            dtxt, l1 = _ask(client, b64, DISTANCE_PROMPT.format(**fmt))
            etxt, l2 = _ask(client, b64, DECISION_PROMPT.format(**fmt))
            lat_total += l1 + l2
            if k == 0:
                first_dtxt = dtxt.strip().replace("\n", " ")[:80]
            d = parse_distance(dtxt)
            e = parse_decision(etxt)
            if d is not None:
                dists.append(d)
            if e is not None:
                execs.append(e)

        if not dists or not execs:
            print(f"  [skip] 파싱 실패 frame={row['frame_id']} raw='{first_dtxt}'")
            continue

        med_dist = float(np.median(dists))
        majority = int(round(np.mean(execs)))
        flip = float(np.mean(np.asarray(execs) != majority))
        writer.writerow({
            "frame_id": row["frame_id"],
            "label": row["label"],
            "vlm_distance_m": round(med_dist, 4),
            "vlm_executable": majority,
            "flip_rate": round(flip, 4),
            "latency_ms": round(lat_total / N_REPEAT, 1),
            "raw_distances": ";".join(f"{x:.3f}" for x in dists),
            "raw_execs": ";".join(str(x) for x in execs),
            "raw_dist_text": first_dtxt,
        })
        f.flush()
        n_written += 1
        print(f"  [{i+1}/{len(rows)}] frame={row['frame_id']} "
              f"dist={med_dist:.2f}m exec={majority} flip={flip:.2f} raw='{first_dtxt}'")
        time.sleep(throttle)

    f.close()
    print(f"저장: {out_path} ({n_written}행)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VLM-only 베이스라인 실행")
    ap.add_argument("--manifest", default=str(DIR / "data" / "manifest.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "vlm_results.csv"))
    a = ap.parse_args()
    run(Path(a.manifest), Path(a.out))
