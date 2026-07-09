"""
Semantic Interpretation Layer 단독 검증 스크립트

실시간 웹캠/비전 파이프라인 없이, 이미 확보된 실험 로그(실험_로그.md)의
실제 측정값 몇 개를 그대로 사용해 Gemini 호출이 의도대로 동작하는지 먼저 확인한다.

⚠️ 이 파일의 샘플은 "새로 만든 가짜 데이터"가 아니라,
실험_로그.md의 log14(phone_hand)/log15(tumbler_hand) 세션에서 실제로 기록된
수치를 그대로 옮긴 것이다. (하드코딩이 아니라 실측 기록 재사용)

⚠️ 이 테스트의 성공 기준에 대한 안내
    이 단계에서 "object_identity가 smartphone/tumbler로 정확히 맞는지"를 성공 기준으로
    삼지 않는다. 입력값(mask_area, target_z 등)만으로는 물병/리모컨/
    스마트폰/손전등 등 여러 사물이 동일하게 설명 가능하므로, 정체성 추론이 매번 다르게
    나올 수 있다. 이는 실패가 아니라 정보 부족에 따른 정상적인 불확실성이다.

    이 단계에서 실제로 확인해야 할 것은 다음 3가지뿐이다:
    1) VLM이 JSON을 안정적으로 반환하는가 (파싱 에러 없이)
    2) results 배열 개수가 입력 objects 개수와 항상 일치하는가
    3) corrected_spatial_relation / semantic_state 값이 매 호출마다 합리적으로 일관되는가

    object_identity의 정확도 검증은 정답 데이터셋을 갖춘 다음 비교 실험(Rule-based vs
    Semantic Layer) 단계에서 별도로 다룬다.

실행:
    export GEMINI_API_KEY="..."
    python -m llm.test_single_object
"""

from llm.schemas import SemanticInterpretationInput, SemanticInterpretationBatchInput
from llm.interpreter import interpret, interpret_batch

# 실험 로그 log14(phone_hand) 세션 실측값: bottle로 오탐지된 스마트폰(손에 든 상태)
SAMPLE_PHONE_HAND = SemanticInterpretationInput(
    object_id=0,
    bbox_2d=[100, 100, 200, 200],
    mask_area=10838,
    centroid_y=404,
    object_x=0.0,
    object_y=0.0,
    target_z=4.86,
    near_distance=5.95,
    floor_depth_delta=1.965,
    context="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중",
)

# 실험 로그 log15(tumbler_hand) 세션 실측값: bottle로 탐지된 텀블러(손에 든 상태)
SAMPLE_TUMBLER_HAND = SemanticInterpretationInput(
    object_id=1,
    bbox_2d=[300, 300, 400, 400],
    mask_area=25535,
    centroid_y=161,
    object_x=0.0,
    object_y=0.0,
    target_z=9.49,
    near_distance=5.49,
    floor_depth_delta=-3.79,
    context="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중",
)


def run_single_call_test():
    print("=" * 60)
    print("[TEST 1] 단일 객체 호출 (interpret) 검증")
    print("=" * 60)

    for name, sample in [("phone_hand", SAMPLE_PHONE_HAND), ("tumbler_hand", SAMPLE_TUMBLER_HAND)]:
        print(f"\n--- 입력: {name} ---")
        print(sample.model_dump_json(indent=2, exclude_none=True))

        output = interpret(sample)

        print(f"\n--- 출력: {name} ---")
        print(output.model_dump_json(indent=2, exclude_none=True))


def run_batch_call_test():
    print("\n" + "=" * 60)
    print("[TEST 2] 배치 호출 (interpret_batch) 검증 — 한 프레임에 2개 객체가 동시에 잡힌 상황 가정")
    print("=" * 60)

    batch_input = SemanticInterpretationBatchInput(
        context="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중",
        objects=[SAMPLE_PHONE_HAND, SAMPLE_TUMBLER_HAND],
    )

    batch_output = interpret_batch(batch_input)

    for input_data, output_data in zip(batch_input.objects, batch_output.results):
        print(
            f"\nobject_id={input_data.object_id} mask_area={input_data.mask_area} "
            f"target_z={input_data.target_z}"
        )
        print(
            f"  -> identity={output_data.identity.class_name} "
            f"is_person={output_data.identity.is_person} "
            f"spatial={output_data.corrected_spatial_relation.environment_relative} "
            f"social={output_data.semantic_state.social_state}"
        )
        print(f"  어포던스: {output_data.semantic_state.affordances}")
        print(f"  행동 정책: {output_data.planner_directives.action_policy}")
        print(f"  애니메이션 트리거: {output_data.planner_directives.animation_trigger}")


if __name__ == "__main__":
    run_single_call_test()
    run_batch_call_test()

    print("\n" + "=" * 60)
    print("[체크리스트] 아래 3가지만 확인하세요 (object_identity 정답 여부는 지금 보지 않음)")
    print("=" * 60)
    print("1) 위 출력이 에러 없이 JSON으로 파싱되었는가?")
    print("2) results 개수가 입력 objects 개수와 일치했는가?")
    print("3) corrected_spatial_relation / semantic_state 값이 합리적으로 일관되는가? (여러 번 실행해서 비교 권장)")
