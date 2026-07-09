"""
Semantic Interpretation Layer 메인 로직

Geometry Layer가 산출한 SemanticInterpretationInput을 받아
Gemini에게 의미 해석을 요청하고 SemanticInterpretationOutput으로 파싱하여 반환한다.
"""

import json
import base64
from io import BytesIO
from typing import Any
from PIL import Image

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL
from .schemas import (
    SemanticInterpretationInput,
    SemanticInterpretationOutput,
    SemanticInterpretationBatchInput,
    SemanticInterpretationBatchOutput,
)

client = OpenAI(api_key=OPENAI_API_KEY)

# VLM에 전달할 이미지 경량화를 기본으로 사용합니다.
# 작은 객체의 시각 정보와 전송량 사이의 균형을 고려해 가로 320px, JPEG 품질 70으로 시작합니다.
# 원본 이미지와 비교 실험하려면 ENABLE_VLM_IMAGE_DOWNSAMPLING을 False로 변경하세요.
ENABLE_VLM_IMAGE_DOWNSAMPLING = True
VLM_IMAGE_MAX_WIDTH = 725
VLM_JPEG_QUALITY = 79


def prepare_vlm_image(image: Image.Image) -> Image.Image:
    """원본 비율을 유지하면서 VLM 전송용 이미지를 축소한다.

    이미 설정한 최대 너비보다 작은 이미지는 선명도 손실을 막기 위해 확대하지 않는다.
    """
    if not ENABLE_VLM_IMAGE_DOWNSAMPLING or image.width <= VLM_IMAGE_MAX_WIDTH:
        return image

    scale = VLM_IMAGE_MAX_WIDTH / image.width
    resized_height = max(1, round(image.height * scale))
    return image.resize(
        (VLM_IMAGE_MAX_WIDTH, resized_height),
        Image.Resampling.LANCZOS,
    )


def encode_image_to_base64(image: Image.Image) -> tuple[str, int]:
    """이미지를 JPEG Base64로 변환하고, 실제 JPEG 바이트 크기도 함께 반환한다."""
    buffered = BytesIO()
    save_options = {}
    if ENABLE_VLM_IMAGE_DOWNSAMPLING:
        # optimize는 JPEG 허프만 테이블을 최적화해 화질을 유지하면서 전송량을 더 줄입니다.
        save_options = {"quality": VLM_JPEG_QUALITY, "optimize": True}
    image.save(buffered, format="JPEG", **save_options)
    jpeg_bytes = buffered.getvalue()
    return base64.b64encode(jpeg_bytes).decode("utf-8"), len(jpeg_bytes)



SYSTEM_PROMPT = """너는 1인칭 단안 RGB 카메라로 촬영된 실내 공간의 원본 이미지와 3D 좌표 기반 객체 수치를 함께 해석하는 공간-의미 분석가다.
목표는 Geometry Layer가 제공한 수치 정보를 그대로 복사하는 것이 아니라, 이미지 증거와 3D 좌표를 함께 검토하여 3D 아바타가 어떤 객체에 어떻게 반응해야 하는지 JSON 스키마에 맞게 결정하는 것이다.

반드시 아래 4단계 판단 흐름을 내부적으로 따른다.

① 장면 배경 서술
- 먼저 카메라 원본 이미지에서 실내 장면의 배경, 주요 표면, 사람/손의 존재, 객체가 놓인 환경을 파악한다.
- 이 단계는 최종 JSON의 별도 필드로 출력하지 말고, 이후 정체성/상태/정책 판단에만 내부적으로 반영한다.

② 객체 정체성 보정
- 입력에는 YOLO 클래스명이나 confidence가 제공되지 않는다. 
- 'class_name'을 판단할 때에는 원본 이미지를 보고 스스로 판단해야 하며 입력 객체 배열의 순서와 1:1로 대응해야 한다.
- 입력 `objects` 배열의 각 항목은 이미 별도의 bbox와 object_id를 가진 독립 객체이다. 일부 객체가 작거나 애매하거나 상호작용 대상이 아니어도 절대 건너뛰지 말고, 모든 object_id에 대해 반드시 정체성을 판단하라.
- 여러 bbox가 같은 물체처럼 보이더라도 임의로 병합하거나 삭제하지 마라. 입력에 6개 객체가 있으면 `results`에도 정확히 6개 객체를 같은 순서로 출력하라.
- 원본 이미지, bbox, 객체 형태, 주변 맥락을 함께 보고 `identity.class_name`을 최종 보정한다.
- 사람 또는 손/신체 일부로 판단되는 경우 `identity.is_person`을 true로 설정한다.

③ 3D 좌표 기반 물리적 상태 추론
- 입력 객체에는 카메라 기준 3D 좌표가 포함된다.
- `object_x`, `object_y`, `target_z`, `floor_depth_delta`, `near_distance`, bbox, 이미지 증거를 함께 보고 객체가 바닥에 있는지, 표면 위에 있는지, 높이 있는지, 떠 있거나 손에 들린 상태인지 추론한다.
- 좌표값은 단안 depth 기반 추정치이므로 절대값 하나만 맹신하지 말고 이미지 증거와 함께 보정한다.

④ 3D 좌표 기반 어포던스 태그 및 행동 정책 부여
- 어포던스는 선택은 3D 좌표, 물체의 높이/거리, 지지면, 형태, 크기, 사용 맥락을 종합하여 가능한 affordance 태그를 최대한 모두 포함하라.
- 구체적인 목적이 있을 때 3D 좌표를 기반으로 목적을 달성하기 위한 어포던스를 최대한 선택해야한다.
- `animation_trigger`는 여러 affordances 중 현재 장면에서 가장 먼저 실행할 단일 행동 하나만 선택한다. 일반적으로 접근 자세나 높이 조정이 필요한 경우 `Bend down and pick up` 또는 `Reach up and take`를 우선 실행 행동으로 선택하고, 단순히 같은 높이에서 잡아 옮기면 형태 기반 태그를 선택한다.
- 사람(`is_person`=true)이거나 객체가 누군가에게 들려 있거나 사용 중이면 접근/상호작용은 안전하지 않다. 이 경우 `action_policy`는 반드시 `IGNORE` 또는 `OBSERVE_ONLY`이고 `is_safe_to_approach=false`이다.
- `environment_relative`가 `on_floor` 또는 `on_surface`이고 `social_state=available`인 경우에만 `APPROACH_AND_INTERACT`를 선택할 수 있다.
- 높거나 멀어서 직접 상호작용이 부적절한 경우 `OBSERVE_ONLY`를 선택한다.

출력 규칙:
1. 응답은 강제된 JSON 스키마를 완벽히 준수해야 하며, 입력 객체 배열의 순서와 1:1로 대응해야 한다.
2. 절대 객체를 생략하지 마라. 상호작용 대상이 아니거나 정체성이 불확실한 객체도 `IGNORE` 또는 `OBSERVE_ONLY`로 판단하여 반드시 `results`에 포함하라.
3. 각 결과의 `object_id`는 입력 객체의 `object_id`를 그대로 유지하라. 결과 순서도 입력 순서와 같아야 한다.
4. 스키마에 없는 추론 설명, 근거 문장, reasoning 필드, 주석, markdown을 절대 출력하지 마라.
5. `semantic_state.affordances`는 반드시 아래 허용 affordance 태그 중 하나 이상을 리스트로 선택한다. 가능한 행동이 여러 개이면 하나로 줄이지 말고 모두 포함한다.
6. `None`은 affordance가 아니라 `planner_directives.animation_trigger`에서만 허용되는 비실행 값이다. 따라서 `semantic_state.affordances`에는 절대 `None`을 넣지 마라.

허용 affordance 태그:
- Spherical grasp to open: 문고리나 둥근 손잡이를 구형 그랩으로 잡고 회전하여 여는 행동
- Wrap grasp to open: 오븐이나 서랍의 바 형태 손잡이를 감싸 쥐고 당겨 여는 행동
- Turn on/off switch: 콘센트나 벽면 스위치 등을 조작하여 전기를 켜거나 끄는 행동
- Press: 리모컨 등의 버튼을 눌러 조작하는 행동
- Two hands raise and move: 큰 그릇이나 무거운 용기를 두 손으로 들어 옮기는 행동
- Cylindrical grasp to move: 머그잔이나 텀블러처럼 원통형 물체를 쥐고 옮기는 행동
- Pinch grasp to move: 종이 타월이나 작은 물건을 손가락 끝으로 집어 옮기는 행동
- Manipulate elongated tools: 칼, 거품기, 국자, 렌치 등 길쭉한 도구를 쥐고 사용하는 행동
- To sit/to place: 사람이 앉거나 물건을 안정적으로 올려놓을 수 있는 지지면을 이용하는 행동
- Bend down and pick up: 바닥이나 낮은 위치에 있는 물체를 사람이 허리를 숙여 집어 올리는 행동
- Reach up and take: 사람보다 높은 위치에 있는 물체를 팔을 위로 뻗어 가져오는 행동
- Observe: 직접 상호작용하지 않고 응시하거나 관찰하는 행동

animation_trigger 규칙:
- `planner_directives.animation_trigger`는 단일 실행 행동 값이다.
- `action_policy=IGNORE`이면 반드시 `animation_trigger=None`으로 설정한다.
- `action_policy=OBSERVE_ONLY`이면 반드시 `animation_trigger=Observe`로 설정하고, `Observe`를 `semantic_state.affordances`에도 포함한다.
- `action_policy=APPROACH_AND_INTERACT`이면 `semantic_state.affordances`에 실제 포함된 상호작용 태그 중 가장 자연스러운 하나를 `animation_trigger`로 선택한다. 이때 `None`이나 `Observe`를 선택하지 마라.

# 예시:
# - 둥근 문고리: `affordances=[Spherical grasp to open, Observe]`, `animation_trigger=Spherical grasp to open`
# - 오븐 손잡이: `affordances=[Wrap grasp to open, Observe]`, `animation_trigger=Wrap grasp to open`
# - 텀블러/물병: `affordances=[Cylindrical grasp to move, Observe]`, `animation_trigger=Cylindrical grasp to move`
# - 바닥의 물병: `affordances=[Bend down and pick up, Cylindrical grasp to move, Observe]`, `animation_trigger=Bend down and pick up`
# - 높은 선반 위 컵: `affordances=[Reach up and take, Cylindrical grasp to move, Observe]`, `animation_trigger=Reach up and take`
# - 멀리 있거나 직접 접근이 부적절한 물체: `affordances=[Observe]`, `animation_trigger=Observe`
# - 사람 또는 손에 들린 물체처럼 무시해야 하는 대상: `affordances=[Observe]`, `animation_trigger=None`
"""


def get_system_prompt() -> str:
    """VLM에 전달할 시스템 프롬프트를 반환한다."""
    return SYSTEM_PROMPT



def interpret(input_data: SemanticInterpretationInput) -> SemanticInterpretationOutput:
    """
    객체 1개를 단독으로 해석한다. (단독 검증/디버깅용. 실시간 다중 객체에는 interpret_batch 사용)
    """
    batch_output = interpret_batch(
        SemanticInterpretationBatchInput(context=input_data.context, objects=[input_data]),
        image=None
    )
    return batch_output.results[0]


def interpret_batch(
    batch_input: SemanticInterpretationBatchInput,
    image: Any = None,
) -> SemanticInterpretationBatchOutput:
    """
    한 프레임에 탐지된 객체 전체를 단 1회의 GPT 호출로 일괄 해석한다.
    이미지(image)가 제공될 경우 멀티모달 프롬프트로 작동한다.
    """
    payload_json = batch_input.model_dump_json()

    user_content = []

    # 이미지가 있으면 Base64로 인코딩하여 GPT-4o Vision 형식에 맞게 추가
    if image is not None:
        original_size = image.size
        vlm_image = prepare_vlm_image(image)
        base64_image, jpeg_size = encode_image_to_base64(vlm_image)
        print(
            f"[VLM 이미지] {original_size[0]}x{original_size[1]} → "
            f"{vlm_image.width}x{vlm_image.height}, JPEG {jpeg_size / 1024:.1f}KB"
        )
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # 텍스트(Geometry/Vision 수치 데이터 JSON) 추가
    user_content.append({
        "type": "text",
        "text": payload_json
    })

    try:
        # OpenAI의 Structured Outputs 기능 사용 (Pydantic 모델 자체를 응답 포맷으로 지정)
        response = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_content}
            ],
            # 스키마의 정의된대로 output 출력이 강제됨.
            response_format=SemanticInterpretationBatchOutput,
            temperature=0.2,
        )
        
        output = response.choices[0].message.parsed
        
    except Exception as e:
        raise RuntimeError(f"OpenAI GPT 호출 또는 파싱에 실패했습니다: {e}") from e

    if len(output.results) != len(batch_input.objects):
        raise RuntimeError(
            f"입력 객체 수({len(batch_input.objects)})와 "
            f"GPT 결과 수({len(output.results)})가 일치하지 않습니다. "
            f"프롬프트의 1:1 대응 규칙을 다시 점검하세요."
        )

    return output
