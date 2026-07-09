"""
Semantic Interpretation Layer (Layer 5) 입출력 스키마

입력: Geometry Layer(Layer 4)가 실시간으로 산출한 scene_data의 객체 1개 분량
출력: LLM(Gemini)이 보정한 사물 정체성/상태/상호작용 가능성
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


AffordanceTag = Literal[
    "Spherical grasp to open",
    "Wrap grasp to open",
    "Turn on/off switch",
    "Press",
    "Two hands raise and move",
    "Cylindrical grasp to move",
    "Pinch grasp to move",
    "Manipulate elongated tools",
    "To sit/to place",
    "Bend down and pick up",
    "Reach up and take",
    "Observe",
]

ActionTrigger = Literal[
    "Spherical grasp to open",
    "Wrap grasp to open",
    "Turn on/off switch",
    "Press",
    "Two hands raise and move",
    "Cylindrical grasp to move",
    "Pinch grasp to move",
    "Manipulate elongated tools",
    "To sit/to place",
    "Bend down and pick up",
    "Reach up and take",
    "Observe",
    "None",
]


class SemanticInterpretationInput(BaseModel):
    """Geometry Layer -> Semantic Interpretation Layer 입력"""

    object_id: int = Field(..., description="객체 식별 ID (이미지 상의 Bounding Box 번호와 매칭됨)")
    bbox_2d: Optional[list[float]] = Field(None, description="YOLO Bounding Box [x1, y1, x2, y2]")
    mask_area: int = Field(..., ge=0, description="SAM 마스크 픽셀 면적")
    centroid_y: int = Field(..., description="마스크 중심점의 y좌표 (픽셀, 화면 세로축)")
    # [Task 5] 최소 요건: 3D 좌표 (X, Y, Z) 전부 + 단위(미터) 명시.
    # 좌표계 = camera_opencv_meters (원점: 카메라 광학중심, +X 우 / +Y 하 / +Z 정면)
    object_x: float = Field(..., description="객체 3D 좌표 X (미터). spatial_3d.x. +는 카메라 오른쪽")
    object_y: float = Field(..., description="객체 3D 좌표 Y (미터). spatial_3d.y. +는 카메라 아래쪽")
    target_z: float = Field(..., description="객체 3D 좌표 Z (미터). spatial_3d.z. 카메라로부터의 정면 거리")
    near_distance: Optional[float] = Field(None, description="가장 가까운 인접 객체와의 거리(m)")
    floor_depth_delta: Optional[float] = Field(
        None, description="바닥 추정 깊이와 객체 깊이의 차이(m) (바닥 접촉 추론 참고용)"
    )
    raw_spatial_guess: Optional[str] = Field(
        None, description="Geometry Layer가 휴리스틱으로 추정한 공간 정보 (예: on_floor, behind_user). 참고용 후보 신호."
    )
    context: str = Field(
        default="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중",
        description="촬영 상황/카메라 시점 맥락",
    )


class Identity(BaseModel):
    class_name: str = Field(..., description="GPT가 이미지와 문맥을 보고 최종 보정한 실제 사물 정체성")
    is_person: bool = Field(..., description="해당 객체가 사람(person) 또는 사람의 신체 일부인지 여부")

class SpatialContext(BaseModel):
    camera_relative: str = Field(..., description="카메라(사용자)와의 상대적 위치 (예: in_front_of_user, far_away)")
    environment_relative: Literal["on_floor", "on_surface", "elevated", "floating", "held"] = Field(
        ..., description="지형지물과의 위상 관계 보정 결과"
    )

class SemanticState(BaseModel):
    social_state: Literal["available", "held_by_user", "in_use_by_other"] = Field(
        ..., description="사물의 점유/사용 상태. 아바타가 건드려도 되는지 판별하는 핵심 정보."
    )
    affordances: list[AffordanceTag] = Field(
        ...,
        min_length=1,
        description="허용 태그 중에서 선택한 하나 이상의 객체 행동 가능성",
    )

class PlannerDirectives(BaseModel):
    action_policy: Literal["APPROACH_AND_INTERACT", "OBSERVE_ONLY", "IGNORE"] = Field(
        ..., description="ActionPlanner를 위한 권고 행동 정책 (권고안일 뿐 최종 명령은 아님)"
    )
    animation_trigger: ActionTrigger = Field(
        ...,
        description="affordances에서 하나를 선택한 단일 실행 행동. 실행하지 않을 때는 None",
    )
    is_safe_to_approach: bool = Field(..., description="위험물이나 타인이 사용중인 물건이 아닌지 여부")

    @model_validator(mode="before")
    @classmethod
    def enforce_policy_trigger(cls, data):
        """비상호작용 정책의 animation trigger를 정책에 맞게 강제한다."""
        if not isinstance(data, dict):
            return data

        normalized = data.copy()
        if normalized.get("action_policy") == "OBSERVE_ONLY":
            normalized["animation_trigger"] = "Observe"
        elif normalized.get("action_policy") == "IGNORE":
            normalized["animation_trigger"] = "None"
        return normalized


class SemanticInterpretationOutput(BaseModel):
    """Semantic Interpretation Layer -> 다음 레이어(Action Planning) 출력 (V2 계층 구조)"""
    
    object_id: int = Field(..., description="입력받은 object_id를 그대로 반환하여 ActionPlanner가 최신 좌표를 매핑하도록 함")
    identity: Identity
    corrected_spatial_relation: SpatialContext
    semantic_state: SemanticState
    planner_directives: PlannerDirectives

    @model_validator(mode="after")
    def validate_action_selection(self):
        """행동 정책, affordance 목록, 단일 trigger 사이의 의미적 일관성을 강제한다."""
        policy = self.planner_directives.action_policy
        trigger = self.planner_directives.animation_trigger

        if trigger != "None" and trigger not in self.semantic_state.affordances:
            self.semantic_state.affordances.append(trigger)
        if policy == "IGNORE" and trigger != "None":
            raise ValueError("IGNORE 정책의 animation_trigger는 None이어야 합니다.")
        if policy == "OBSERVE_ONLY" and trigger != "Observe":
            raise ValueError("OBSERVE_ONLY 정책의 animation_trigger는 Observe여야 합니다.")
        if policy == "APPROACH_AND_INTERACT":
            if trigger in ("None", "Observe"):
                raise ValueError("APPROACH_AND_INTERACT에는 상호작용 trigger가 필요합니다.")

        return self


class SemanticInterpretationBatchInput(BaseModel):
    """한 프레임에 탐지된 객체 여러 개를 한 번에 GPT에 전달하기 위한 컨테이너"""

    context: str
    objects: list[SemanticInterpretationInput]


class SemanticInterpretationBatchOutput(BaseModel):
    """GPT가 한 번에 반환하는 객체별 결과 리스트"""

    results: list[SemanticInterpretationOutput]
