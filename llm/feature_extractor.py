"""
Geometry Layer가 생성한 scene_data(실시간 측정값)에서
Semantic Interpretation Layer 입력을 추출한다.

scene_data 구조 (기존 vision 파이프라인 출력 기준):
{
    "scene": {...},
    "objects": [
        {
            "label": "bottle",
            "yolo": {"bbox_2d": [...]},
            "sam": {"mask_area": 10838, "centroid_2d": [612, 404]},
            "depth": {"mean_relative_depth": ...},
            "spatial_3d": {"x": ..., "y": ..., "z": 4.86},
            "near_distance": 5.95,          # SpatialRelationGraph가 채워줌 (없을 수 있음)
            "floor_depth_delta": 1.965,
        },
        ...
    ]
}
"""

from typing import Optional

from .schemas import SemanticInterpretationInput

DEFAULT_CONTEXT = "1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중"


def build_input_from_object(
    obj: dict,
    object_id: int,
    context: str = DEFAULT_CONTEXT,
) -> SemanticInterpretationInput:
    """
    scene_data["objects"]의 객체 1개(dict)를 받아
    SemanticInterpretationInput으로 변환한다. (실시간 값 그대로 사용, 하드코딩 없음)
    """
    return SemanticInterpretationInput(
        object_id=object_id,
        bbox_2d=obj.get("yolo", {}).get("bbox_2d"),
        mask_area=obj["sam"]["mask_area"],
        centroid_y=obj["sam"]["centroid_2d"][1],
        object_x=obj["spatial_3d"]["x"],
        object_y=obj["spatial_3d"]["y"],
        target_z=obj["spatial_3d"]["z"],
        near_distance=obj.get("near_distance"),
        floor_depth_delta=obj.get("floor_depth_delta"),
        context=context,
    )


def build_inputs_from_scene(
    scene_data: dict,
    context: str = DEFAULT_CONTEXT,
) -> list[SemanticInterpretationInput]:
    """scene_data 한 프레임 전체에서 객체별 입력 리스트를 추출한다."""
    inputs = []
    for idx, obj in enumerate(scene_data.get("objects", [])):
        inputs.append(build_input_from_object(obj, idx, context=context))
    return inputs
