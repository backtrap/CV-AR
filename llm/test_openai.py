import sys
import asyncio
from PIL import Image

from llm.schemas import SemanticInterpretationBatchInput, SemanticInterpretationInput
from llm.interpreter import interpret_batch

def test_openai_integration():
    print("1. 테스트용 더미 데이터 생성 중...")
    
    # 100x100 짜리 검은색 더미 이미지 생성 (OpenAI Vision 테스트용)
    dummy_image = Image.new('RGB', (100, 100), color = 'black')
    
    # YOLO가 'bottle'을 찾았다고 가정한 더미 입력
    test_input = SemanticInterpretationBatchInput(
        context="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중",
        objects=[
            SemanticInterpretationInput(
                object_id=0,
                bbox_2d=[10.0, 10.0, 50.0, 50.0],
                mask_area=1500,
                centroid_y=30,
                object_x=0.0,
                object_y=0.0,
                target_z=1.5,
                near_distance=None,
                floor_depth_delta=None,
                context="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중"
            ),
            SemanticInterpretationInput(
                object_id=1,
                bbox_2d=[60.0, 10.0, 90.0, 90.0],
                mask_area=3000,
                centroid_y=50,
                object_x=0.0,
                object_y=0.0,
                target_z=2.0,
                near_distance=0.5,
                floor_depth_delta=None,
                context="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중"
            )
        ]
    )
    
    print("2. OpenAI GPT-4o-mini 호출 중 (Structured Outputs 테스트)...")
    try:
        result = interpret_batch(test_input, image=dummy_image)
        print("\n[성공] OpenAI 응답을 정상적으로 파싱했습니다!")
        print("="*50)
        for obj in result.results:
            print(f"객체 ID: {obj.object_id}")
            print(f"정체성: {obj.identity.class_name} (사람 여부: {obj.identity.is_person})")
            print(f"공간 위치: {obj.corrected_spatial_relation.camera_relative}, {obj.corrected_spatial_relation.environment_relative}")
            print(f"상태: {obj.semantic_state.social_state}")
            print(f"어포던스: {obj.semantic_state.affordances}")
            print(f"행동 정책: {obj.planner_directives.action_policy} (안전 여부: {obj.planner_directives.is_safe_to_approach})")
            print(f"애니메이션 트리거: {obj.planner_directives.animation_trigger}")
            print("-" * 30)
    except Exception as e:
        print(f"\n[실패] 에러 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_openai_integration()
