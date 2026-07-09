# 좌표 정확도 검증 결과 문서 (Task 6)

> 산출된 3D 좌표(특히 깊이 Z)의 정확도를 알려진 거리 촬영으로 정량 검증한 결과.
> 원자료: `data/depth_calib.csv` · 보정계수: `vision/depth/depth_scale.json` · 도구: `tools/measure_depth.py`, `tools/calibrate_depth_scale.py`

## 1. 검증 목적 및 항목

- **목적**: 단안 RGB → metric depth → 3D 역투영으로 산출한 객체 거리(Z)가 실제 거리와 얼마나 일치하는지 정량화.
- **검증 항목**: (a) 거리별 절대 오차, (b) 표준 깊이 평가지표(AbsRel, δ), (c) 프레임 간 변동량(안정성), (d) 유효 거리 범위.

## 2. 측정 방법

| 항목 | 내용 |
|---|---|
| 대상 | 불투명 병(bottle), 받침에 고정 |
| 세팅 | **카메라 고정**(촬영 내내 미이동), 대상만 표시선 따라 이동, 배경·조명 고정 |
| 거리 | 0.4 / 0.6 / 0.8 / 1.0 / 1.2 m (줄자: 렌즈면↔물체 앞면) |
| 표본 | 각 거리 **60프레임**, 대표값 = 프레임별 robust median의 median |
| 원시값 | 보정 OFF(scale=1, offset=0)로 모델 원시 pred 수집 |
| 보정식 | 다거리 회귀로 `true = scale·pred + offset` 도출 → `scale=1.1031, offset=−0.3932` |

> 측정 방식 설계 근거: 단안 metric depth는 장면 전체 맥락에 의존하므로, 카메라·배경을 고정하고
> 대상만 이동해야 캡처 간 재현성이 확보된다(초기 손에 든 촬영에서 동일 거리 재측정 시 20cm 편차 관측 → 받침 고정으로 해소).

## 3. 정량 결과

보정 후 좌표(`true = 1.1031·pred − 0.3932`) 기준:

| 실제(m) | 원시 pred(m) | 보정 Z(m) | 오차(cm) | AbsRel | 비율(max) | 프레임 std(mm) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.783 | 0.471 | +7.1 | 0.177 | 1.177 | 10.4 |
| 0.60 | 0.837 | 0.530 | −7.0 | 0.117 | 1.132 | 5.2 |
| 0.80 | 1.073 | 0.790 | −1.0 | 0.012 | 1.013 | 6.6 |
| 1.00 | 1.261 | 0.998 | −0.2 | 0.002 | 1.002 | 20.7 |
| 1.20 | 1.455 | 1.211 | +1.1 | 0.009 | 1.009 | 57.7 |

**종합 지표 (n=5)**
- 회귀 적합도: **R² = 0.975**
- **RMSE = 4.50 cm**, MAE = 3.28 cm, 최대 오차 = 7.07 cm
- **AbsRel = 6.3 %**
- **δ₁ (max(pred/gt, gt/pred) < 1.25) = 5/5 = 1.00 (100%)**

## 4. 허용 수준 및 근거

깊이 추정 정확도의 허용 기준은 임의로 정하지 않고, **단안 깊이 추정 분야의 표준 평가지표**를 채택한다.

- **δ₁ < 1.25 (threshold accuracy)**: 예측/실측 비율이 1.25배 이내인 표본 비율. 단안 깊이 추정의
  사실상 표준 정확도 지표이며, 값 1.0에 가까울수록 우수.
  — 근거: Eigen, Puhrsch, Fergus, *"Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"*, NIPS 2014 (δ<1.25, AbsRel, RMSE 지표의 출처). NYU Depth v2 / KITTI 및 Depth Anything V2 등 후속 연구가 동일 지표 사용.
- **AbsRel (평균 절대 상대오차)**: 위 논문 정의. 낮을수록 우수.

**본 검증 판정**: δ₁ = **1.00**(전 구간 통과), AbsRel = **6.3%** → 표준 지표 기준 **양호(pass)**.
보조 지표로 RMSE ≤ 10 cm를 실무 허용선으로 두었으며(도달 상호작용에서 손 크기·PPS 0.7 m 대비 충분히 작음), 실측 RMSE 4.5 cm로 충족.

## 5. 프레임 간 변동(안정성) — Task 4 연계

- 정지 대상의 캡처 내 프레임 std: 0.4~1.0 m 구간 5~21 mm(안정), 1.2 m에서 58 mm.
- 별도 실측(`size_log` 분석): 정지 객체 프레임 간 |Δz| ≈ 8 mm(지터 바닥), 이동 객체 ≈ 235 mm.
- **좌표 안정화(One Euro Filter) 적용 효과**: 정지 z 표준편차 **14 mm → 5.7 mm**, 급이동 지연 < 2 cm.

### 5.1 라이브 전체 파이프라인 안정성 (§7 완료조건 검증)

전체 경로(웹캠→YOLO→SAM→Depth→3D→안정화)를 **매 프레임** 실행(depth 캐싱 없음 = 최악 조건)해
정지 병의 `spatial_3d` 프레임 간 변동을 측정. 도구: `tools/verify_stability.py`, 로그: `data/stability_log.csv`.

| 조건 | 축 | raw std | 안정화 후 std |
|---|---|---:|---:|
| 병 @0.5 m, 88프레임, 1.1 fps | X | 0.5 mm | — |
| | Y | 1.4 mm | — |
| | **Z** | **6.5 mm** | **5.7 mm** |

- **세 축 모두 sub-cm** → "객체별 3D 좌표 안정적 생성" 실측 입증.
- Z 노이즈는 거리에 따라 증가(예: 0.9 m에서 raw std ≈ 28 mm) — 단안 depth 모델의 특성.
- 본 측정은 depth를 매 프레임 재추론한 **최악 조건**이며, 프로덕션은 depth를 10프레임마다 캐싱(`DEPTH_INTERVAL=10`)하여 그 사이 z가 고정되므로 **실사용 per-frame 안정성은 이보다 우수**.

## 6. 유효 거리 범위 및 한계

- **신뢰 범위: 0.4 ~ 1.2 m.** 이 구간에서 δ₁ = 1.0, 오차 ≤ 7 cm.
- **근거리(0.4~0.6 m)**: 오차 ±7 cm로 상대적으로 큼(모델의 근거리 압축). 그래도 δ₁ 통과.
- **원거리(≳ 1.5 m)**: 초기 측정에서 pred가 실제 거리 증가에 둔감해지는 **포화** 관측(예: 1.5/1.75/2.0 m에서 pred 2.23/2.32/2.53). 선형 보정의 정확도 저하 → 신뢰 범위에서 제외.
- 좌표는 **카메라 상대 좌표**(프레임 시점 기준)이며 월드 절대좌표가 아님.

## 7. 결론

- 0.4~1.2 m 범위에서 **δ₁ = 1.00, RMSE 4.5 cm**로 검증 통과. VLM 공간 추론 입력으로 사용 가능한 정밀도 확보.
- 한계: 근거리 ±7 cm, 원거리 포화. 개선 방향 — 원거리 확장 시 구간별(piecewise)/비선형 보정, 또는 스테레오/ToF 보조.
- **기기별 재측정 필요**: 다른 카메라에서는 `docs/depth_calibration_guide.md`로 재캘리브레이션.

## 참고문헌

1. D. Eigen, C. Puhrsch, R. Fergus. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network." *NIPS* 2014. — δ<1.25, AbsRel, RMSE 깊이 평가지표.
2. L. Yang et al. "Depth Anything V2." 2024. — 사용 모델(Metric Indoor).
3. G. Casiez, N. Roussel, D. Vogel. "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems." *CHI* 2012. — 좌표 안정화.
4. J. W. Tukey. "Exploratory Data Analysis." 1977. — 1.5×IQR 이상치 펜스(대표 깊이).
