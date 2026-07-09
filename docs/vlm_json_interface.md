# Vision ↔ VLM JSON 인터페이스 명세 (공유용 v1)

> **대상**: VLM(Semantic Interpretation) 파트 담당자
> **목적**: Vision(Geometry) 파트가 넘기는 입력 JSON과 VLM이 돌려줄 출력 JSON의 **계약(contract)** 을 확정.
> **정의 위치**: `llm/schemas.py` (pydantic). 이 문서는 그 스키마와 1:1로 대응.
> 필드/타입은 스키마가 단일 진실원본(SoT)이며, 변경 시 이 문서도 함께 갱신.

## 0. 데이터 흐름

```
[Vision 파이프라인] → SemanticInterpretationBatchInput  (JSON, 아래 §2)
                          │
                          ▼  (interpret_batch → GPT/VLM)
[VLM 해석]        → SemanticInterpretationBatchOutput (JSON, 아래 §3)
                          │
                          ▼
[ActionPlanner 등 다음 레이어]
```

- 한 프레임에 탐지된 객체 여러 개를 **배치로** 한 번에 주고받는다.
- 매칭 키는 `object_id` (입력의 id를 출력이 그대로 반환 → 좌표 재매핑용).

## 1. 좌표계 (입력의 3D 좌표에 공통 적용)

- **원점**: 카메라 광학 중심. **단위**: 미터(m).
- **축(OpenCV 오른손)**: `+X` = 오른쪽, `+Y` = 아래쪽, `+Z` = 카메라 정면(피사체 방향).
- 값 출처: 각 객체의 `spatial_3d.{x,y,z}` (SAM 중심 픽셀 + metric depth 역투영, depth는 기기별 캘리브레이션 보정 적용됨).

---

## 2. 입력: Vision → VLM  (`SemanticInterpretationBatchInput`)

### 최상위
| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `context` | string | 선택 | 촬영 상황/시점 맥락 |
| `objects` | array\<Object\> | 필수 | 객체별 입력 리스트 |

### objects[] (`SemanticInterpretationInput`)
| 필드 | 타입 | 필수 | 단위 | 설명 |
|---|---|---|---|---|
| `object_id` | int | ✅ | - | 프레임 내 객체 식별 ID (출력이 그대로 반환) |
| `detected_class` | string | ✅ | - | YOLO 탐지 클래스명 (**오탐 가능** — VLM이 보정) |
| `confidence` | float | ✅ | 0~1 | YOLO 신뢰도 |
| `object_x` | float | ✅ | **m** | 3D 좌표 X (+오른쪽) |
| `object_y` | float | ✅ | **m** | 3D 좌표 Y (+아래) |
| `target_z` | float | ✅ | **m** | 3D 좌표 Z (카메라 정면 거리) |
| `mask_area` | int | ✅ | px | SAM 마스크 픽셀 면적 |
| `centroid_y` | int | ✅ | px | 마스크 중심 y(화면 세로) |
| `bbox_2d` | [float×4] | 선택 | px | YOLO 박스 [x1,y1,x2,y2] |
| `near_distance` | float\|null | 선택 | m | 최근접 인접 객체와의 거리 |
| `floor_depth_delta` | float\|null | 선택 | m | 바닥 추정깊이 − 객체깊이 (바닥 접촉 추론용) |
| `raw_spatial_guess` | string\|null | 선택 | - | Geometry의 휴리스틱 공간 추정(참고 신호) |
| `context` | string | 선택 | - | 객체 단위 맥락(기본값 있음) |

> **최소 요건(과제 확정)**: 식별정보(`detected_class`,`confidence`) + 3D 좌표(`object_x`,`object_y`,`target_z`, 미터) 는 항상 존재.

### 입력 예시
```json
{
  "context": "1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중",
  "objects": [
    {
      "object_id": 0,
      "bbox_2d": [612.0, 340.0, 690.0, 470.0],
      "detected_class": "bottle",
      "confidence": 0.41,
      "mask_area": 10838,
      "centroid_y": 404,
      "object_x": -0.12,
      "object_y": 0.03,
      "target_z": 0.70,
      "near_distance": 0.55,
      "floor_depth_delta": 0.18,
      "raw_spatial_guess": null,
      "context": "1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중"
    }
  ]
}
```

---

## 3. 출력: VLM → 다음 레이어  (`SemanticInterpretationBatchOutput`)

### 최상위
| 필드 | 타입 | 설명 |
|---|---|---|
| `results` | array\<Output\> | 객체별 해석 결과 |

### results[] (`SemanticInterpretationOutput`)
| 필드 | 타입 | 설명 |
|---|---|---|
| `object_id` | int | 입력 id 그대로 반환(좌표 재매핑용) |
| `identity` | object | 정체성 보정 |
| `corrected_spatial_relation` | object | 공간 관계 보정 |
| `semantic_state` | object | 점유/상호작용 상태 |
| `planner_directives` | object | 플래너 권고 |
| `reasoning` | string | 판단 이유(한국어 1문장 내외) |

**identity**: `class_name`(string, 최종 보정 정체성), `is_person`(bool)
**corrected_spatial_relation**: `camera_relative`(string, 예 `in_front_of_user`), `environment_relative`(enum: `on_floor`\|`on_surface`\|`elevated`\|`floating`\|`held`)
**semantic_state**: `social_state`(enum: `available`\|`held_by_user`\|`in_use_by_other`), `affordances`(array\<AffordanceTag\>, 1개 이상)
**planner_directives**: `action_policy`(enum: `APPROACH_AND_INTERACT`\|`OBSERVE_ONLY`\|`IGNORE`), `animation_trigger`(AffordanceTag 1개), `is_safe_to_approach`(bool)

### AffordanceTag 허용값 (닫힌 집합)
```
"Spherical grasp to open", "Wrap grasp to open", "Turn on/off switch",
"Press", "Two hands raise and move", "Cylindrical grasp to move",
"Pinch grasp to move", "Manipulate elongated tools", "To sit/to place",
"Observe", "None"
```

### 일관성 규칙 (스키마가 강제 — 위반 시 검증 실패)
- `action_policy == OBSERVE_ONLY` → `animation_trigger`는 반드시 `"Observe"`
- `action_policy == IGNORE` → `animation_trigger`는 반드시 `"None"`
- `action_policy == APPROACH_AND_INTERACT` → `animation_trigger`는 `None`/`Observe`가 아닌 실제 상호작용 태그
- `animation_trigger`가 `affordances`에 없으면 자동 추가됨

### 출력 예시
```json
{
  "results": [
    {
      "object_id": 0,
      "identity": { "class_name": "water bottle", "is_person": false },
      "corrected_spatial_relation": {
        "camera_relative": "in_front_of_user",
        "environment_relative": "on_surface"
      },
      "semantic_state": {
        "social_state": "available",
        "affordances": ["Wrap grasp to open", "Cylindrical grasp to move"]
      },
      "planner_directives": {
        "action_policy": "APPROACH_AND_INTERACT",
        "animation_trigger": "Cylindrical grasp to move",
        "is_safe_to_approach": true
      },
      "reasoning": "책상 위 사용 가능한 물병으로, 손 닿는 거리라 접근·상호작용 권장."
    }
  ]
}
```

---

## 4. 참고

- 호출 진입점: `llm/interpreter.py :: interpret_batch(batch_input)` — 입력을 `model_dump_json()`으로 직렬화해 VLM에 전달.
- 입력 생성: `llm/feature_extractor.py :: build_inputs_from_scene(scene_data)` — Vision의 `scene_data`에서 위 입력을 추출.
- 거리 관련 필드(`target_z`,`near_distance`,`floor_depth_delta`)는 **기기별 depth 캘리브레이션**을 거친 값. 신뢰 범위 ≈0.4~1.2m (그 밖은 오차 증가) — `docs/depth_calibration_guide.md` 참고.
- 스키마 변경 시: `llm/schemas.py` 수정 → 이 문서 동기화 → VLM 파트에 공지.
