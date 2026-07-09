# Object Representation 인터페이스 & 좌표 검증 명세 (v1)

> Vision(Geometry) 파트 → VLM(Semantic) 파트 전달 문서.
> Task 3~6의 확정 사항과 근거를 정리한다.

## 0. 위임 판단 결정 요약 (§4)

담당자에게 위임된 판단 항목과 그 결정·근거·상세위치. (판단 결과는 모두 아래 절 및 코드에 문서화됨)

| # | 위임 판단 항목 | 결정 | 근거(요약) | 상세 |
|---|---|---|---|---|
| 1 | 객체 대표 **위치** 계산 방식 | 마스크 픽셀 **무게중심** | 경계오차 영향 ∝ 둘레/면적 = 1/√area → 큰 마스크에서 자동 상쇄. SAM 경계도 매끈 | §3.1 |
| 2 | 객체 대표 **깊이** 계산 방식 | 유효성필터(≥0.05m)+**Tukey 1.5×IQR**+**median** | 분포가정 없는 robust 이상치 제거(Tukey 1977). mean은 배경 tail에 끌림 | §3.2 |
| 3 | **좌표계** 정의 | `camera_opencv_meters` (원점=광학중심, +X우/+Y하/+Z정면, m) | OpenCV 오른손 관례, 핀홀 역투영과 정합 | §1 |
| 4 | 좌표 **안정화** 적용 여부·방법 | **적용**. One Euro Filter, beta=10, 라벨+IoU 매칭 | 정지지터8mm vs 이동235mm(30배)→속도적응형. beta는 실측 스윕으로 결정 | §5 |
| 5 | Object Representation **형식·필드명** | **JSON**, `SemanticInterpretationInput` 스키마 | pydantic 검증·직렬화 용이, VLM 배치 입력 | §2, `vlm_json_interface.md` |

## 1. 좌표계 정의 (Task 3)

- **원점**: 카메라 광학 중심(pinhole).
- **단위**: 미터(m). Z는 metric depth, X/Y는 역투영으로 동일 스케일.
- **축(OpenCV 오른손 관례)**: `+X` = 이미지 오른쪽, `+Y` = 이미지 아래쪽, `+Z` = 카메라 정면(피사체 방향).
- **역투영식**: `X = (u - cx)·Z / fx`, `Y = (v - cy)·Z / fy`
- **내부 파라미터**: `vision/stream.py`의 캘리브레이션된 `CAMERA_MATRIX`
  (fx≈964.86, fy≈964.45, cx≈636.84, cy≈359.35 @1280×720). 파이프라인이 이를 주입한다.
- `scene["coordinate_system"]` 값: `"camera_opencv_meters"`.

## 2. Object Representation 인터페이스 (Task 5)

VLM 입력 스키마: `llm/schemas.py :: SemanticInterpretationInput`.
최소 요건(식별정보·3D 좌표·단위) 충족 필드:

| 필드 | 타입 | 단위 | 설명 |
|---|---|---|---|
| `object_id` | int | - | 프레임 내 객체 식별 ID |
| `detected_class` | str | - | YOLO 클래스명(오탐 가능) |
| `confidence` | float | - | YOLO 신뢰도 0~1 |
| `object_x` | float | **m** | 3D 좌표 X (+오른쪽) |
| `object_y` | float | **m** | 3D 좌표 Y (+아래) |
| `target_z` | float | **m** | 3D 좌표 Z (정면 거리) |
| `mask_area` | int | px | SAM 마스크 면적 |
| `centroid_y` | int | px | 마스크 중심 y |
| `near_distance` | float? | m | 최근접 객체 거리 |
| `floor_depth_delta` | float? | m | 바닥-객체 깊이차 |

> 원 좌표(spatial_3d)에는 `raw_xyz`(안정화 전), `stabilized`, `distance_from_agent`,
> `dimensions_cm`, `corners`도 포함된다.

## 3. 대표값 산출 방식과 이상치 제거 근거

### 3.1 대표 픽셀 (u, v) — Task 1
**방식**: SAM 마스크에 속한 모든 픽셀의 무게중심 `(mean(x), mean(y))`.
`vision/segmentation/segmenter.py :: segment_objects`.

**경계 노이즈에 강한 이유(검토 결과)**:
- 마스크 경계 오차는 **둘레(perimeter, O(√area))** 픽셀에만 영향을 주는 반면, 무게중심은
  **면적(area) 전체** 픽셀 평균이다. 경계가 대표점에 미치는 상대 영향은 대략 `둘레/면적 ∝ 1/√area`로,
  객체 마스크가 클수록(수천~수만 px) 급격히 작아진다. 즉 무게중심 자체가 경계 노이즈를 평균으로 상쇄한다.
- 또한 SAM은 YOLO bbox 프롬프트 기반이라 경계가 비교적 매끈해 경계 픽셀 비중이 애초에 낮다.

**결정(위임 항목)**: erosion(경계 침식)이나 마스크 median 좌표 같은 추가 강건화는 위 이유로 실익이 작아
**무게중심을 채택**한다. (마스크가 매우 작거나 파편화된 극단 케이스에서는 erosion/median이 후보 — 향후 개선.)

### 3.2 대표 깊이 Z — Task 2  ★근거 있는 이상치 제거
`vision/depth/depth_estimator.py :: robust_representative_depth`
1. **유효성 필터**: 유한하고 `≥ 0.05m`(실내 팔길이 하한, 모델 노이즈 컷) 픽셀만.
2. **Tukey 1.5×IQR 펜스**: `[Q1-1.5·IQR, Q3+1.5·IQR]` 밖 제거.
   - 임계값이 **상수가 아니라 그 객체 분포(Q1,Q3)에서 자동 도출** → 데이터 기반 근거.
   - 마스크 경계가 문 배경 tail을 제거.
3. **대표값 = inlier median** (평균과 달리 50% 오염까지 강건).
4. 표본 < 20이면 펜스 없이 median만(과소표본 방어).
- 검증: 객체0.7m + 배경3.0m 20% 오염 시 평균 1.16m(오염) → 강건 median 0.70m.
- 산출물에 `method`, `inlier_ratio`, `n_samples` 를 남겨 추적 가능.

### 3.3 Depth 스케일 계수 — Task 6  ★근거 있는 값으로 전환
- 기존 `0.51`은 **단일 측정(22cm/43cm), 스케일만** → 근거 부족.
- 현재: `vision/depth/depth_scale.json`에서 `true_m = scale·pred + offset` 로드
  (파일에 provenance/n_samples/R²/RMSE 기록). 미존재 시 0.51 폴백.
- 도구: `tools/calibrate_depth_scale.py` — 다거리 측정을 회귀해 scale·offset·R²·RMSE 산출 후 파일 갱신.
  - scale-only vs scale+offset 자동 비교(RMSE 작은 쪽 채택).
- **캘리브레이션 완료**: `depth_scale.json` = `scale=1.1031, offset=−0.3932` (R²=0.975, RMSE=4.5cm).
  기기별 재측정은 `docs/depth_calibration_guide.md` 참조.

## 4. 좌표 정확도 검증 (Task 6) — 완료

측정·검증 상세 결과는 **`docs/coordinate_accuracy_validation_result.md`** 참조.

- **방법**: bottle 받침 고정, 카메라 고정, 0.4~1.2m 5점 ×60프레임, 원시 pred 수집(`tools/measure_depth.py`) → 회귀.
- **결과**: δ₁(<1.25) = **1.00**, AbsRel = **6.3%**, RMSE = **4.5cm**, 최대오차 7.1cm.
- **허용 기준 근거**: 단안 깊이 표준지표 δ₁<1.25 (Eigen et al., NIPS 2014). δ₁=1.0으로 통과.
- **신뢰 범위 0.4~1.2m** (≳1.5m는 모델 포화로 제외).
- **프레임 간 변동**: 안정화(1€ 필터) 적용으로 정지 z_std 14mm→5.7mm.

## 5. 좌표 안정화 (Task 4)

`vision/spatial/stabilizer.py :: CoordinateStabilizer` (One Euro Filter, Casiez et al. 2012).
- 객체 연결: 라벨 + bbox IoU(≥0.3) greedy 매칭(YOLO id는 프레임 내 인덱스라 시간축 정체성 없음).
- **파라미터 근거**: 실측 정지 지터 ≈8mm/frame vs 이동 ≈235mm/frame(30배차) → 속도적응형 채택.
  미터 단위 스케일 보정 위해 `beta=10` (스윕 결과: 정지지터 2.5mm, 보행1m/s 지연 1.5cm).
- 효과: 정지 z_std 14mm → 5.7mm, 급이동 지연 <2cm. `raw_xyz`로 원본 보존.
- **최종 beta 확정은 4장 정지 측정으로 검증 예정.**
