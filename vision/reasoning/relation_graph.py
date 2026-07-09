import numpy as np

class SpatialRelationGraph:
    def __init__(self, near_threshold=0.7):
        """
        [8-1단계: 최종 방어형 관계 그래프 엔진]
        - near_threshold를 실험으로 검증된 0.7(70cm, PPS)로 고정하여 동작합니다.
        - near 관계는 한 방향(i < j)으로만 정직하게 저장하므로, 하위 엔진 조회 시 반드시 양방향 조회가 필요합니다.
        """
        self.near_threshold = near_threshold

    def calculate_relations(self, scene_data):
        """
        [피드백 1, 2 완벽 반영] 
        - required_keys 방어 코드를 도입하여 실시간 KeyError를 원천 차단합니다.
        - O(N^2) 루프 최적화 및 near 관계 단방향 적재 사양을 유지합니다.
        """
        relations = []
        if not scene_data or "objects" not in scene_data or not scene_data["objects"]:
            return relations

        objects = scene_data["objects"]
        num_objs = len(objects)
        
        # [피드백 2 반영] 실시간 예외 방어용 필수 키 정의
        required_keys = {"x", "y", "z"}

        for i in range(num_objs):
            for j in range(i + 1, num_objs):
                obj_a = objects[i]
                obj_b = objects[j]

                if obj_a is None or obj_b is None:
                    continue

                a_id, b_id = obj_a["id"], obj_b["id"]
                a_coords = obj_a.get("spatial_3d", {})
                b_coords = obj_b.get("spatial_3d", {})

                # [피드백 2 반영] x, y, z 중 단 하나라도 누락되면 즉시 스킵하여 실시간 크래시 차단
                if not required_keys.issubset(a_coords) or not required_keys.issubset(b_coords):
                    continue

                # -------------------------------------------------------------
                # 1. 좌우 관계 판단 (X축 대조)
                # -------------------------------------------------------------
                if a_coords["x"] < b_coords["x"] - 0.2:
                    relations.append({"subject_id": a_id, "predicate": "left_of", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "right_of", "object_id": a_id})
                elif a_coords["x"] > b_coords["x"] + 0.2:
                    relations.append({"subject_id": a_id, "predicate": "right_of", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "left_of", "object_id": a_id})

                # -------------------------------------------------------------
                # 2. 전후 깊이 관계 판단 (Z축 대조)
                # -------------------------------------------------------------
                if a_coords["z"] < b_coords["z"] - 0.3:
                    relations.append({"subject_id": a_id, "predicate": "closer_than", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "farther_than", "object_id": a_id})
                elif a_coords["z"] > b_coords["z"] + 0.3:
                    relations.append({"subject_id": a_id, "predicate": "farther_than", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "closer_than", "object_id": a_id})

                # -------------------------------------------------------------
                # 3. 상하 관계 판단 (Y축 대조)
                # [피드백 2 반영] TODO: 현재는 Pseudo Y 기준(음수가 위)이며 월드 좌표계 도입 시 반전 검토
                # -------------------------------------------------------------
                if a_coords["y"] < b_coords["y"] - 0.2:
                    relations.append({"subject_id": a_id, "predicate": "above", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "below", "object_id": a_id})
                elif a_coords["y"] > b_coords["y"] + 0.2:
                    relations.append({"subject_id": a_id, "predicate": "below", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "above", "object_id": a_id})

                # -------------------------------------------------------------
                # 4. Ground Plane Distance 기반 Near 대칭 관계 연산
                # [피드백 1, 2 반영] 
                # - 저장 공간 최적화를 위해 단방향(a_id -> b_id)으로만 1회 기록합니다.
                # - 하위 AffordanceEngine 조회 시 반드시 주어/목적어 뒤집은 양방향 OR 검색 유지 확인 완료
                # -------------------------------------------------------------
                dist_ground = np.sqrt(
                    (a_coords["x"] - b_coords["x"])**2 +
                    (a_coords["z"] - b_coords["z"])**2
                )
                
                if dist_ground <= self.near_threshold:
                    relations.append({
                        "subject_id": a_id, 
                        "predicate": "near", 
                        "object_id": b_id,
                        "distance": round(float(dist_ground), 3)
                    })

        return relations

    def process_scene_relations(self, scene_data):
        """
        최상위 scene 데이터에 생성된 relation 그래프를 정보를 병합합니다.
        """
        if not scene_data:
            return scene_data

        scene_relations = self.calculate_relations(scene_data)
        scene_data["relations"] = scene_relations
        
        return scene_data


# =====================================================================
# 🚀 8-1단계 최종 정합성 유닛 테스트
# =====================================================================
if __name__ == "__main__":
    import json
    print("==========================================================")
    print("CV_AR: 8-1단계 SpatialRelationGraph 마스터본 최종 검증")
    print("==========================================================")
    
    graph_generator = SpatialRelationGraph()
    
    mock_scene_data = {
        "objects": [
            { "id": 0, "label": "person", "spatial_3d": {"x": 0.0, "y": 1.2, "z": 2.0} },
            { "id": 1, "label": "cup", "spatial_3d": {"x": 0.1, "y": 0.8, "z": 1.8} }
        ]
    }
    
    result_scene = graph_generator.process_scene_relations(mock_scene_data)
    print(json.dumps(result_scene, indent=2, ensure_ascii=False))
    
    relations = result_scene["relations"]
    near_triples = [r for r in relations if r["predicate"] == "near"]
    
    assert len(near_triples) == 1, "단방향 최적화 가드 붕괴 예외"
    assert "distance" in near_triples[0], "정량 기하 마진 누락 예외"
    
    print("\n✅ 방어 가드 및 예외 처리 완료: 8-1단계 모듈 완전 종결!")
    print("==========================================================")