# Depth 스케일 캘리브레이션 가이드 (팀원용)

> **이 작업은 노트북(웹캠)마다 따로 해야 합니다.**
> depth 보정 계수(`depth_scale.json`)는 "그 웹캠에서 depth 모델이 거리를 얼마나 틀리게 보는지"를
> 담은 값이라, 기기가 바뀌면 값도 달라집니다. 새 노트북에서 파이프라인을 쓰기 전에 아래를 1회 수행하세요.
> 소요 시간 약 10분.

---

## 0. 준비

- 저장소 클론 + venv 활성화 + 의존성 설치 완료된 상태 (`torch`, `ultralytics`, `transformers`)
- **작동하는 웹캠**
- **줄자**
- **불투명한 대상 물체 1개** — 병/상자/책 등. YOLO(COCO)가 잡는 것. **투명·반사 물체 금지.**
- **대상을 올려둘 받침/탁자** (아주 중요 — 아래 촬영 원칙 참고)

먼저 프로젝트 루트로 이동:
```bash
cd <저장소 경로>/CV-AR
source .venv/bin/activate        # 이미 됐으면 생략
```

---

## 1. 촬영 원칙 (이거 안 지키면 값이 튑니다)

monocular depth는 **장면 전체 맥락**으로 절대거리를 추정합니다. 그래서:

1. **카메라를 고정하고 촬영 내내 절대 건드리지 마세요.** (삼각대/책 받침 등)
2. **대상 물체만** 바닥의 표시선을 따라 앞뒤로 옮깁니다. 배경·조명은 그대로.
3. **손에 들지 마세요.** 받침에 올려두세요. (손에 들면 캡처마다 20cm씩 값이 튐)
4. 대상이 화면에 **크고 정면으로** 잡히게.

---

## 2. 거리별 측정

**시작 전 기존 데이터 비우기** (남의 노트북/이전 측정값이 섞이지 않게):
```bash
rm -f data/depth_calib.csv
```

각 거리마다 아래를 한 번씩 실행. 대상 라벨은 예시로 `bottle`(자기 물체에 맞게 변경).
줄자는 **렌즈 면 ↔ 물체 앞면** 기준.

```bash
python3 -m tools.measure_depth --true 0.4 --label "bottle"
python3 -m tools.measure_depth --true 0.6 --label "bottle"
python3 -m tools.measure_depth --true 0.8 --label "bottle"
python3 -m tools.measure_depth --true 1.0 --label "bottle"
python3 -m tools.measure_depth --true 1.2 --label "bottle"
```

각 실행:
- 창 상단 `detected:`에 대상 라벨이 뜨고, **초록 박스+마스크**가 물체를 덮고, `[n/60]` 카운터가 올라가면 정상.
- 60프레임 채우면 자동 종료·저장. 마지막 줄에 `✅ ... 1행 추가/덮어씀` 확인.
- **끝 출력의 `std`(mm)를 보세요. 30mm 넘으면 그 거리만 다시 실행** (같은 거리는 덮어써짐).

> 라벨을 모르면 `--label` 빼고 한 번 띄워서 `detected:`에 뭐가 잡히는지 확인 후 그 이름을 사용.
> 결과는 `data/depth_calib.csv`에 누적됩니다.

---

## 3. 회귀 → 품질 확인

```bash
python3 -m tools.calibrate_depth_scale data/depth_calib.csv
```

출력에서 **모델 B** 기준으로 아래 합격 기준을 확인:

| 항목 | 합격 기준 |
|---|---|
| R² | ≥ 0.97 |
| RMSE | ≤ 10cm (낮을수록 좋음, 목표 ~5cm) |
| 거리별 잔차 | 특정 점만 크게 튀지 않을 것 |
| 단조성 | 거리 늘수록 pred도 커질 것(줄어드는 점 있으면 그 점 재측정) |

**기준 미달이면** 잔차 큰 거리 / std 높았던 거리만 다시 측정(2번) 후 이 단계 반복.

---

## 4. 확정 (파일 갱신)

기준을 만족하면:
```bash
python3 -m tools.calibrate_depth_scale data/depth_calib.csv --write
```
→ `vision/depth/depth_scale.json`이 갱신됩니다.

적용 확인:
```bash
python3 -c "from vision.depth.depth_estimator import load_depth_scale; print(load_depth_scale())"
```
`(scale, offset)` 튜플이 나오면 파이프라인이 이 값을 자동으로 씁니다. 끝.

---

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `[경고] 수집된 pred 없음` | 대상이 화면에 안 잡힘 → 조명 밝게, 물체 크게/정면, 라벨 확인 |
| 특정 거리만 std 200mm↑ | 손에 들었거나 흔들림 → 받침에 올리고 재측정 |
| 같은 거리 재측정마다 값이 20cm씩 다름 | 카메라가 움직임 → 카메라 완전 고정 후 전 거리 재측정 |
| 1.5m 넘어가면 잔차 급증 | 모델 포화. 신뢰 범위를 ~1.2m로 제한 |
| pred가 거리 따라 안 늘어남 | 대상이 투명/반사 → 불투명 물체로 교체 |

---

## 참고: 신뢰 범위 & 기기별 재현

- 위 절차로 얻은 계수는 **측정한 거리 범위(예: 0.4~1.2m) 안에서만 신뢰**됩니다. 그 밖은 외삽이라 오차 큼.
- `depth_scale.json`은 **커밋하지 말고 각자 로컬에만** 두는 걸 권장(기기마다 다르므로).
  공유 시엔 누구 노트북 값인지 명시.
- (고급) 카메라 내부 파라미터 `CAMERA_MATRIX`(`vision/stream.py`)도 기기별입니다.
  정밀도가 더 필요하면 `vision/calibration.py`로 체커보드 캘리브레이션을 따로 수행해 교체하세요.
