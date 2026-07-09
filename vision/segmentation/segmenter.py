import cv2
import numpy as np
import os
import torch
from ultralytics import SAM
import json
import copy

# 하위 패키지 모듈 참조 정합성 유지
from vision.stream import WebcamStream, CAMERA_MATRIX
from vision.detector import ObjectDetector
from vision.depth.depth_estimator import DepthEstimator, robust_representative_depth
from vision.spatial.transformer import Spatial3DConverter
from vision.spatial.stabilizer import CoordinateStabilizer
from vision.spatial.floor_detector import FloorPlaneDetector
from vision.reasoning.relation_graph import SpatialRelationGraph   # ➔ 8단계 관계 그래프 엔진 수입
from vision.reasoning.affordance_engine import AffordanceEngine # ➔ 8단계 어포던스 추론 엔진 수입

# [최적화 스케줄링 상수]
SAM_INTERVAL = 5     # SAM 연산은 5프레임 당 한 번
DEPTH_INTERVAL = 10  

class ObjectSegmenter:
    def __init__(self):
        self.model = SAM("sam_b.pt")
        
        # Windows + NVIDIA GPU면 CUDA 사용
        # 하드웨어 자동 감지 예외 처리 (M4 맥북 등)
        if torch.cuda.is_available():
            self.device = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            gpu_name = "Apple MPS"
        else:
            self.device = "cpu"
            gpu_name = "CPU"
        
        print(f"SAM 모델이 [{self.device}] 가속 엔진 위에서 성공적으로 로드되었습니다.")

    def segment_objects(self, frame, scene_data):
        """
        annotated_frame, scene_data, masks_list을 반환,
        각각 초록색으로 물체가 칠해진 화면 출력용 이미지, SAM 필드가 채워진 json,
        SAM이 만든 객체별 마스크들을 따로 모아둔 리스트 (프레임 크기와 동일한 bool type 배열)
        => 깊이를 계산할때 사용
        """
        
        if not scene_data or "objects" not in scene_data or not scene_data["objects"]:
            return frame, scene_data, []

        mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        masks_list = []

        all_bboxes = [obj["yolo"]["bbox_2d"] for obj in scene_data["objects"] if obj is not None]
        if not all_bboxes:
            return frame, scene_data, []
            
        results = self.model(frame, bboxes=all_bboxes, device=self.device, verbose=False)
        result_masks = results[0].masks

        if result_masks is None:
            return frame, scene_data, []

        for idx, obj in enumerate(scene_data["objects"]):
            if idx >= len(result_masks.data):
                break
                
            mask_bool = result_masks.data[idx].cpu().numpy().astype(bool)
            mask_pixels = int(np.sum(mask_bool))
            
            y_indices, x_indices = np.where(mask_bool)
            cx = int(np.mean(x_indices)) if len(x_indices) > 0 else 0
            cy = int(np.mean(y_indices)) if len(y_indices) > 0 else 0

            obj["sam"] = {
                "mask_path": None,
                "mask_area": mask_pixels,
                "centroid_2d": [cx, cy]
            }
            masks_list.append(mask_bool)

            mask_overlay[mask_bool] = [0, 255, 0]
            cv2.circle(mask_overlay, (cx, cy), 5, (0, 0, 255), -1)

        annotated_frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.4, 0)
        return annotated_frame, scene_data, masks_list

    def overlay_cached_masks(self, frame, scene_data, cached_sam_data):
        """
        SAM을 새로 돌리거나 마스크 통계를 다시 계산하지 않고,
        이전 SAM 실행에서 완성된 객체별 sam 정보를 현재 scene_data에 합칩니다.

        cached_sam_data의 각 항목:
        {"mask_path": ..., "mask_area": ..., "centroid_2d": [...]}
        """

        for idx, obj in enumerate(scene_data["objects"]):
            if idx >= len(cached_sam_data):
                break

            # 현재 객체가 이후 단계에서 값을 변경해도 캐시 원본이 오염되지 않도록 복사합니다.
            obj["sam"] = copy.deepcopy(cached_sam_data[idx])

        return frame, scene_data


class SceneDepthAttacher:
    """
    json 데이터에 깊이 정보도 attach 하는 클래스
    """
    def __init__(self):
        pass

    def attach_depth(self, scene_data, masks_list, depth_data):
        if not scene_data or "objects" not in scene_data or not scene_data["objects"] or not masks_list or depth_data is None:
            return scene_data

        raw_depth_map = depth_data["raw_depth"]
        if len(raw_depth_map.shape) == 3:
            raw_depth_map = np.squeeze(raw_depth_map)
            
        h, w = raw_depth_map.shape[:2]

        for idx, obj in enumerate(scene_data["objects"]):
            if obj is None or not isinstance(obj, dict) or idx >= len(masks_list):
                continue
                
            mask_bool = masks_list[idx]
            if "sam" not in obj or "mask_area" not in obj["sam"]: continue
            mask_pixels = obj["sam"]["mask_area"]

            if mask_pixels == 0:
                obj["depth"] = robust_representative_depth(None)
                continue

            if raw_depth_map.shape != mask_bool.shape:
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            roi_pixels = raw_depth_map[mask_bool]
            # [Task2] 단순 평균 대신 IQR 이상치 제거 후 median (근거: robust_representative_depth 참조)
            obj["depth"] = robust_representative_depth(roi_pixels)

        return scene_data


# =====================================================================
# 🚀 메인 제어 엔진룸 (YOLO ➔ SAM ➔ Depth ➔ 3D ➔ Floor ➔ Graph ➔ Affordance 대통합)
# =====================================================================
if __name__ == "__main__":
    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth_engine = DepthEstimator()
    attacher = SceneDepthAttacher()
    # [Task3 결함1 수정] 캘리브레이션된 내부 파라미터(CAMERA_MATRIX) 주입.
    # 기본값(fx=900) 대신 stream.py의 실측 행렬(fx≈964.86)을 사용해 좌표 오차를 줄인다.
    spatial_converter = Spatial3DConverter(CAMERA_MATRIX)
    stabilizer = CoordinateStabilizer()  # [Task4] 프레임 간 좌표 안정화(1€ 필터)
    floor_detector = FloorPlaneDetector()
    relation_graph = SpatialRelationGraph() # ➔ 8단계-1 관계 생성기 장착
    affordance_engine = AffordanceEngine()   # ➔ 8단계-2 의미 추론기 장착
    
    # 순서조정 : 맨 마지막에 웹캠 열기
    stream = WebcamStream()

    frame_count = 0
    cached_depth_data = None 
    
    last_masks_list = []
    last_sam_data = []
    last_labels_sequence = []

    print("==========================================================")
    print("CV_AR: [8단계 어포던스 추론 연동] 대통합 파이프라인 실시간 가동")
    print("-> 30프레임마다 최상위 relations 그래프와 객체별 actions가 인쇄됩니다.")
    print("==========================================================")

    while True:
        ret, frame = stream.get_frame()
        if not ret:
            break
            
        frame_count += 1
        
        # 1. Depth 연산 스케줄링 (10프레임 주기)
        if frame_count == 1 or frame_count % DEPTH_INTERVAL == 0:
            cached_depth_data = depth_engine.get_depth_map(frame)
        
        # 2. YOLO 객체 탐지 (results[0] 패치 반영 완료)
        yolo_result = detector.detect(frame)
        current_scene = detector.build_scene(yolo_result, frame, frame_count)
        current_labels = [obj["label"] for obj in current_scene["objects"] if obj is not None]

        # 캐시 정합성 고속 디스패처
        # 캐싱 조건 1 : 5프레임마다 SAM 돌리는 데 5프레임 아직 안지나고
        # 캐싱 조건 2 : 이전 객체 갯수가 현재 갯수와 똑같고
        # 캐싱 조건 3 : 이전 객체 라벨의 순서가 현재와 똑같으면
        can_use_cache = (
            frame_count % SAM_INTERVAL != 0 and 
            len(last_masks_list) == len(current_scene["objects"]) and
            current_labels == last_labels_sequence
        )
        
        # SAM mask 캐싱이 가능한 조건이면 이전 프레임의 SAM 정보 사용
        if can_use_cache:
            annotated_frame, mid_scene = segmenter.overlay_cached_masks(frame, current_scene, last_sam_data)
            scene_with_depth = attacher.attach_depth(mid_scene, last_masks_list, cached_depth_data)
            scene_with_3d = spatial_converter.process_scene_3d(scene_with_depth)
        # 조건이 안맞으면 SAM을 통해 새로운 추론 진행
        else:
            annotated_frame, mid_scene, last_masks_list = segmenter.segment_objects(frame, current_scene)
            last_sam_data = [copy.deepcopy(obj.get("sam")) for obj in mid_scene["objects"]]
            scene_with_depth = attacher.attach_depth(mid_scene, last_masks_list, cached_depth_data)
            scene_with_3d = spatial_converter.process_scene_3d(scene_with_depth)
            last_labels_sequence = copy.deepcopy(current_labels)

        # 2.5 [Task4] 좌표 안정화: 관계/어포던스 추론 전에 spatial_3d 지터 제거
        scene_with_3d = stabilizer.process_scene(scene_with_3d)

        # 3. 7단계 가설면 매핑 레이어 주입
        scene_with_floor = floor_detector.update_scene_with_floor(scene_with_3d, cached_depth_data)

        # 4. [8단계-1 핵심 이식] 실시간 3D 기하학 수치 기반 위상 관계 트리플 그래프 빌드
        scene_with_relations = relation_graph.process_scene_relations(scene_with_floor)

        # 5. [8단계-2 핵심 이식] 라벨 속성 + 공간 상태 융합형 동적 어포던스(actions) 최종 추론
        final_scene = affordance_engine.infer_affordances(scene_with_relations)

        # 30프레임마다 결과 출력
        if frame_count % 30 == 0:
            print(f"\n--- [FRAME {frame_count}] 30fps 실시간 8단계 지능형 장면 그래프 완료 ---")
            print(json.dumps(final_scene, ensure_ascii=False, indent=2))
            
        cv2.imshow("CV_AR - Golden Master Unified Pipeline", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    stream.release()
    cv2.destroyAllWindows()
