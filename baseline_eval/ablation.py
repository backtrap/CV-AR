"""
Self-ablation 'Ours − 기하(PPS) 엔진' 독립 구현.

본문(vision/reasoning/affordance_engine.py)을 import 하지 않고, 그 정적
property_registry / property_to_action을 여기에 '미러링(복사)'해 둔다.
→ baseline_eval 폴더가 본문과 완전히 분리된 상태로 ablation을 계산.

ablation 정의:
  - 본문 어포던스 엔진 = 정적 property_registry + 거리(PPS 0.7m) 게이트.
  - 'Ours − 기하' = 거리 게이트를 제거한 버전. 물체가 실행가능 property를
    가지면 거리와 무관하게 항상 executable=1.
  - 전체 Ours와의 차이 = PPS 기하의 순수 기여(= 논문 핵심 ablation).

⚠ 아래 두 딕셔너리는 본문 affordance_engine.py의 값을 그대로 복사한 것이다.
   본문 레지스트리가 바뀌면 여기도 함께 갱신할 것(현재 기준: bottle 등 COCO 15종).
"""

from __future__ import annotations

# --- 본문 affordance_engine.py 미러 (복사본, import 아님) ---------------------
PROPERTY_REGISTRY = {
    "person": ["interactive", "communicative"],
    "chair": ["sittable", "movable"],
    "desk": ["placeable_on", "leanable_on"],
    "table": ["placeable_on", "leanable_on"],
    "cup": ["graspable", "drinkable", "placeable_on"],
    "wine glass": ["graspable", "drinkable"],
    "bottle": ["graspable", "drinkable", "openable"],
    "cell phone": ["graspable", "usable"],
    "laptop": ["graspable", "usable", "openable"],
    "book": ["graspable", "readable", "openable"],
    "remote": ["graspable", "usable"],
    "handbag": ["graspable", "openable", "carryable"],
    "backpack": ["graspable", "openable", "carryable"],
    "suitcase": ["graspable", "movable", "openable"],
    "umbrella": ["graspable", "openable"],
}

PROPERTY_TO_ACTION = {
    "graspable": "grasp",
    "usable": "use",
    "drinkable": "drink",
    "readable": "read",
    "openable": "open",
    "sittable": "sit",
    "movable": "move",
    "carryable": "carry",
}
# -----------------------------------------------------------------------------

# 실행가능(interactive) 액션으로 이어지는 property 집합. 거리 게이트 없이 판단.
ACTIONABLE = set(PROPERTY_TO_ACTION) | {"placeable_on", "leanable_on"}


def actions_for_label(label: str) -> list[str]:
    """거리 무시하고 라벨이 지원하는 실행 액션 목록(정적)."""
    props = PROPERTY_REGISTRY.get(label, ["inspectable"])
    acts = [PROPERTY_TO_ACTION[p] for p in props if p in PROPERTY_TO_ACTION]
    acts += [p.replace("able", "") for p in props if p in {"placeable_on", "leanable_on"}]
    return acts


def executable_no_geometry(label: str) -> int:
    """Ours − 기하: 물체가 실행가능 property를 가지면 거리와 무관하게 1."""
    props = PROPERTY_REGISTRY.get(label, ["inspectable"])
    return int(any(p in ACTIONABLE for p in props))


def ablation_executable(labels) -> list[int]:
    """프레임 라벨 리스트 → executable(0/1) 리스트."""
    return [executable_no_geometry(lb) for lb in labels]


if __name__ == "__main__":
    for lb in ["bottle", "chair", "table", "person", "unknown_thing"]:
        print(f"{lb:16s} exec(no-geo)={executable_no_geometry(lb)}  "
              f"actions={actions_for_label(lb)}")
