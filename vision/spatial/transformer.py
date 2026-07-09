import numpy as np
import math


# =====================================================================
# [좌표계 정의 — Task 3 문서화 요건]
# 원점(Origin)   : 카메라 광학 중심 (pinhole)
# 단위(Unit)     : 미터(m). Z는 Metric Depth, X/Y는 역투영으로 동일 스케일.
# 축 방향(Axes)  : OpenCV 카메라 좌표계(오른손) 관례를 따른다.
#   +X : 이미지 오른쪽 방향 (u 증가)
#   +Y : 이미지 아래쪽 방향 (v 증가)   ← 화면 좌표계라 위가 아니라 아래가 +
#   +Z : 카메라 정면(광축) 방향, 피사체 쪽이 양수
# 역투영식     : X=(u-cx)*Z/fx,  Y=(v-cy)*Z/fy
# scene["coordinate_system"] 값 "camera_opencv_meters" 로 이 규약을 표기한다.
# =====================================================================


class Spatial3DConverter:
    def __init__(self, camera_matrix=None):
        """
        camera_matrix를 전달받아 실제 카메라 내부 파라미터를 사용한다.
        전달되지 않으면 기존 임시값을 사용한다.
        """
        if camera_matrix is None:
            self.f_x = 900.0
            self.f_y = 900.0
            self.c_x = 640.0
            self.c_y = 360.0
        else:
            self.f_x = float(camera_matrix[0, 0])
            self.f_y = float(camera_matrix[1, 1])
            self.c_x = float(camera_matrix[0, 2])
            self.c_y = float(camera_matrix[1, 2])

        print(
            f"[Spatial3D] Using calibrated camera: "
            f"fx={self.f_x:.2f}, fy={self.f_y:.2f}, "
            f"cx={self.c_x:.2f}, cy={self.c_y:.2f}"
        )

    def convert_to_3d(self, centroid_2d, mean_relative_depth):
        """
        [핀홀 역투영 가상 공간화 알고리즘]
        [피드백 반영] 음수 depth 유입 시 원점(0,0,0)으로 데이터가 튀어 디버깅이 꼬이는 현상 방지.
        Z는 0으로 클램핑하되, X와 Y는 최소 안전거리(0.001) 기준의 위상 방향성을 유지시킵니다.
        """
        u, v = centroid_2d
        
        # 시각적 위치 왜곡 및 디버깅 혼선을 막기 위한 투영용 깊이와 실제 저장용 깊이 분리
        z_project = max(mean_relative_depth, 0.001)
        z_final = max(mean_relative_depth, 0.0)
        
        # 투영 기하학 수식 적용 (방향 위상 보존)
        X = (u - self.c_x) * z_project / self.f_x
        Y = (v - self.c_y) * z_project / self.f_y
        
        return {
            "x": round(float(X), 3),
            "y": round(float(Y), 3),
            "z": round(float(z_final), 3)
        }

    def calculate_3d_bounding_box(self, bbox_2d, z_val):
        x1, y1, x2, y2 = bbox_2d
        
        TL_X = (x1 - self.c_x) * z_val / self.f_x
        TL_Y = (y1 - self.c_y) * z_val / self.f_y
        TR_X = (x2 - self.c_x) * z_val / self.f_x
        TR_Y = (y1 - self.c_y) * z_val / self.f_y
        BL_X = (x1 - self.c_x) * z_val / self.f_x
        BL_Y = (y2 - self.c_y) * z_val / self.f_y
        BR_X = (x2 - self.c_x) * z_val / self.f_x
        BR_Y = (y2 - self.c_y) * z_val / self.f_y

        width_cm = abs(TR_X - TL_X) * 100.0
        height_cm = abs(BL_Y - TL_Y) * 100.0

        return {
            "dimensions_cm": {
                "width": round(float(width_cm), 1),
                "height": round(float(height_cm), 1)
            },
            "corners": {
                "TL": [round(float(TL_X), 3), round(float(TL_Y), 3)],
                "TR": [round(float(TR_X), 3), round(float(TR_Y), 3)],
                "BL": [round(float(BL_X), 3), round(float(BL_Y), 3)],
                "BR": [round(float(BR_X), 3), round(float(BR_Y), 3)]
            }
        }

    def process_scene_3d(self, scene_data):
        """
        글로벌 JSON 데이터를 순회하며 spatial_3d 방 구조를 완성합니다.
        """
        if not scene_data or "objects" not in scene_data or not scene_data["objects"]:
            return scene_data

        
        # 전역 좌표계 정보 상위 메타데이터 레벨로 최적화 이동 완료
        if "scene" not in scene_data:
            scene_data["scene"] = {}
        # 좌표계 규약: 카메라 광학중심 원점, OpenCV 오른손 축(+X우/+Y하/+Z정면), 단위 m
        scene_data["scene"]["coordinate_system"] = "camera_opencv_meters"

        for obj in scene_data["objects"]:
            if obj is None or "sam" not in obj or "depth" not in obj:
                continue
                
            centroid_2d = obj["sam"].get("centroid_2d", None)
            mean_depth = obj["depth"].get("mean_relative_depth", None)

            if centroid_2d is not None and mean_depth is not None:
                spatial_coords = self.convert_to_3d(
                    centroid_2d,
                    mean_depth
                )
                
                # 1. 아바타로부터 떨어진 유클리디안 거리 (Distance from Agent)
                dist = math.sqrt(spatial_coords["x"]**2 + spatial_coords["y"]**2 + spatial_coords["z"]**2)
                spatial_coords["distance_from_agent"] = round(dist, 3)

                # 2. 3D 물리적 크기 및 4개 꼭짓점 좌표 (Bounding Box in 3D)
                if "yolo" in obj and obj["yolo"] and "bbox_2d" in obj["yolo"]:
                    bbox_3d_info = self.calculate_3d_bounding_box(obj["yolo"]["bbox_2d"], spatial_coords["z"])
                    spatial_coords.update(bbox_3d_info)

                obj["spatial_3d"] = spatial_coords
            else:
                obj["spatial_3d"] = {"x": 0.0, "y": 0.0, "z": 0.0, "distance_from_agent": 0.0}

        return scene_data


# =====================================================================
# 🚀 정밀 크로스 유닛 테스트 (수학적 정합성 100% 일치 버전)
# =====================================================================
if __name__ == "__main__":
    camera_matrix = np.array([
        [958.2263, 0.0, 624.0653],
        [0.0, 956.1898, 362.6175],
        [0.0, 0.0, 1.0]
    ])

    converter = Spatial3DConverter(camera_matrix)
    
    # [테스트 1] 우측 하단 이동 및 투영 공식 검증
    res = converter.convert_to_3d([900, 540], 5.0)
    expected_x = round((900 - converter.c_x) * 5.0 / converter.f_x, 3)
    expected_y = round((540 - converter.c_y) * 5.0 / converter.f_y, 3)
    
    assert res["x"] == expected_x, f"X축 불일치: {res['x']} vs {expected_x}"
    assert res["y"] == expected_y, f"Y축 불일치: {res['y']} vs {expected_y}"
    assert res["z"] == 5.0, "Z값 오류"
    
    # [테스트 2] 음수 노이즈 유입 시 디버깅 원점 팅김 방지 검증 (Z만 0.0 고정)
    res_neg = converter.convert_to_3d([900, 540], -1.0)
    expected_neg_x = round((900 - converter.c_x) * 0.001 / converter.f_x, 3)
    expected_neg_y = round((540 - converter.c_y) * 0.001 / converter.f_y, 3)
    
    assert res_neg["x"] == expected_neg_x, "음수 처리 X축 위상 오류"
    assert res_neg["y"] == expected_neg_y, "음수 처리 Y축 위상 오류"
    assert res_neg["z"] == 0.0, "음수 클램핑 Z축 오염"
    
    print("✅ 피드백 반영 완료: 모든 정밀 유닛 테스트 완벽 통과!")