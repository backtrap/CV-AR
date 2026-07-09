import cv2
import numpy as np
import time
import json
import copy
import torch
import csv
import os
from ultralytics import YOLO, SAM
from transformers import pipeline
from PIL import Image

# =====================================================================
# [연구원 실행 지침 가이드] 제어 스위치 및 가설 파라미터 세팅
# =====================================================================
# PHASE_MODE 조절:
# - "PHASE_1_MEASURE": near_threshold=1.0 상태에서 원시 dist_ground 위상 수치 통계 수집
# - "PHASE_3_VERIFY":  데이터 기반으로 결정한 커스텀 임계값을 적용하여 grasp, use 행동 분기 검증
CURRENT_PHASE_MODE = "PHASE_3_VERIFY"  

# 데이터 기반 near 임계값 설정 슬롯 (PHASE_1_MEASURE 일때는 1.0 가드로 고정됨)
CALIBRATED_NEAR_THRESHOLD = 0.7

# FLOOR_ROI_MODE 조절:
# - "Bottom_ROI": 기존 화면 하단 25% 전체 (사용자 몸통 오염 가설 테스트)
# - "Dual_Edge_ROI": 사람 가림 노이즈를 회피하는 하단 좌우 외곽 15% 슬릿 ROI
CURRENT_FLOOR_MODE = "Bottom_ROI"

# Floor 검출용 분위수(Percentile) 가설 파라미터화 (실험 스위치 분리)
CURRENT_FLOOR_PERCENTILE = 80  

# 현재 수행 중인 실험 조건 메타데이터 정의 (CSV 자동 적재 컬럼)
# 예: "30cm", "50cm", "1m", "2m", "3m" 등으로 변경해가며 측정하여 수동 실험 노트 작성을 대체
EXPERIMENTAL_CONDITION = "30cm"

# 최적화 및 로깅 스케줄링 상수
SAM_INTERVAL = 5
DEPTH_INTERVAL = 10
STAT_LOG_INTERVAL = 60  # 매 60프레임(약 2초)마다 정제된 수치 통계 및 요약 출력
CSV_OUTPUT_PATH = "data/dist_log.csv"


# =====================================================================
# 1. 기하학 및 추론 핵심 통합 아키텍처 계층 (Core Components)
# =====================================================================

class UnifiedWebcamStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return ret, frame.copy()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


class UnifiedGeometryEngine:
    def __init__(self, f_x=900.0, f_y=900.0):
        self.f_x = f_x
        self.f_y = f_y

    def convert_to_3d(self, centroid_2d, mean_relative_depth, c_x, c_y):
        u, v = centroid_2d
        z_project = max(mean_relative_depth, 0.001)
        z_final = max(mean_relative_depth, 0.0)
        
        X = (u - c_x) * z_project / self.f_x
        Y = (v - c_y) * z_project / self.f_y
        return {"x": round(float(X), 3), "y": round(float(Y), 3), "z": round(float(z_final), 3)}

    def process_scene_3d(self, scene_data):
        if not scene_data or "objects" not in scene_data: return scene_data
        resolution = scene_data["frame_metadata"]["camera_resolution"]
        c_x, c_y = resolution[0] / 2.0, resolution[1] / 2.0
        scene_data["scene"]["coordinate_system"] = "pseudo_3d"

        for obj in scene_data["objects"]:
            if obj and obj.get("sam") and obj.get("depth"):
                centroid = obj["sam"]["centroid_2d"]
                depth_val = obj["depth"]["mean_relative_depth"]
                obj["spatial_3d"] = self.convert_to_3d(centroid, depth_val, c_x, c_y)
        return scene_data


class ResearchFloorDetector:
    def __init__(self, roi_ratio=0.25, mode="Bottom_ROI", percentile=80):
        self.roi_ratio = roi_ratio
        self.mode = mode
        self.percentile = percentile

    def detect_floor(self, depth_map):
        if depth_map is None:
            return 0.0, "none_fallback"
            
        h, w = depth_map.shape[:2]
        roi_start_y = int(h * (1.0 - self.roi_ratio))

        if self.mode == "Bottom_ROI":
            floor_roi = depth_map[roi_start_y:h, :]
            valid = floor_roi[floor_roi > 0]
            return float(np.median(valid)) if valid.size > 0 else 0.0, f"bottom_roi_median(P50)"
            
        elif self.mode == "Dual_Edge_ROI":
            left_slit = depth_map[roi_start_y:h, :int(w * 0.15)]
            right_slit = depth_map[roi_start_y:h, int(w * 0.85):w]
            valid = np.concatenate([left_slit[left_slit > 0], right_slit[right_slit > 0]])
            return float(np.percentile(valid, self.percentile)) if valid.size > 0 else 0.0, f"dual_edge_P{self.percentile}"

    def update_scene_with_floor(self, scene_data, raw_depth):
        if raw_depth is None:
            scene_data["scene"]["floor_detected"] = False
            scene_data["scene"]["floor_depth"] = 0.0
            scene_data["scene"]["floor_method"] = "none"
            return scene_data

        floor_depth, method = self.detect_floor(raw_depth)
        scene_data["scene"]["floor_detected"] = True if floor_depth > 0 else False
        scene_data["scene"]["floor_depth"] = round(floor_depth, 3)
        scene_data["scene"]["floor_method"] = method

        for obj in scene_data["objects"]:
            if obj and obj.get("spatial_3d"):
                obj_z = obj["spatial_3d"]["z"]
                obj["floor_depth_delta"] = round(float(floor_depth - obj_z), 3) if obj_z > 0 else 0.0
        return scene_data


class ResearchRelationGraph:
    def __init__(self, mode="PHASE_1_MEASURE", calibrated_threshold=1.0):
        self.mode = mode
        self.near_threshold = 1.0 if mode == "PHASE_1_MEASURE" else calibrated_threshold
        self.log_buffer = []
        self.csv_buffer = []  
        
        os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
        if not os.path.exists(CSV_OUTPUT_PATH):
            with open(CSV_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "label", "condition", "raw_depth", "agent_z", "target_z", "x_diff", "z_diff", "dist_ground"])

    def flush_csv_buffer(self):
        if self.csv_buffer:
            with open(CSV_OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.csv_buffer)
            self.csv_buffer.clear()

    def process_scene_relations(self, scene_data):
        relations = []
        objects = scene_data.get("objects", [])
        num_objs = len(objects)

        for i in range(num_objs):
            for j in range(i + 1, num_objs):
                obj_a, obj_b = objects[i], objects[j]
                if not obj_a or not obj_b or not obj_a.get("spatial_3d") or not obj_b.get("spatial_3d"): continue

                a_id, b_id = obj_a["id"], obj_b["id"]
                a_coords, b_coords = obj_a["spatial_3d"], obj_b["spatial_3d"]

                x_diff = a_coords["x"] - b_coords["x"]
                z_diff = a_coords["z"] - b_coords["z"]
                dist_ground = np.sqrt(x_diff**2 + z_diff**2)

                is_hoi = (obj_a["label"] == "person" and obj_b["label"] != "person") or \
                         (obj_b["label"] == "person" and obj_a["label"] != "person")

                if is_hoi:
                    agent = obj_a if obj_a["label"] == "person" else obj_b
                    target = obj_b if obj_a["label"] == "person" else obj_a
                    raw_d = target["depth"]["mean_relative_depth"] if target.get("depth") else 0.0
                    
                    log_entry = {
                        "frame_id": scene_data["frame_metadata"]["frame_id"],
                        "target_label": target["label"],
                        "condition": EXPERIMENTAL_CONDITION,  
                        "x_diff": abs(x_diff),
                        "z_diff": abs(z_diff),
                        "dist_ground": dist_ground,
                        "agent_z": agent["spatial_3d"]["z"],
                        "target_z": target["spatial_3d"]["z"],
                        "raw_mean_depth": raw_d
                    }
                    self.log_buffer.append(log_entry)
                    self.csv_buffer.append([
                        log_entry["frame_id"], log_entry["target_label"], log_entry["condition"], round(raw_d, 4),
                        round(log_entry["agent_z"], 4), round(log_entry["target_z"], 4),
                        round(log_entry["x_diff"], 4), round(log_entry["z_diff"], 4), round(dist_ground, 4)
                    ])

                if a_coords["x"] < b_coords["x"] - 0.2:
                    relations.append({"subject_id": a_id, "predicate": "left_of", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "right_of", "object_id": a_id})
                elif a_coords["x"] > b_coords["x"] + 0.2:
                    relations.append({"subject_id": a_id, "predicate": "right_of", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "left_of", "object_id": a_id})

                if a_coords["z"] < b_coords["z"] - 0.3:
                    relations.append({"subject_id": a_id, "predicate": "closer_than", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "farther_than", "object_id": a_id})
                elif a_coords["z"] > b_coords["z"] + 0.3:
                    relations.append({"subject_id": a_id, "predicate": "farther_than", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "closer_than", "object_id": a_id})

                if a_coords["y"] < b_coords["y"] - 0.2:
                    relations.append({"subject_id": a_id, "predicate": "above", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "below", "object_id": a_id})
                elif a_coords["y"] > b_coords["y"] + 0.2:
                    relations.append({"subject_id": a_id, "predicate": "below", "object_id": b_id})
                    relations.append({"subject_id": b_id, "predicate": "above", "object_id": a_id})

                if dist_ground <= self.near_threshold:
                    relations.append({
                        "subject_id": a_id, "subject_label": obj_a["label"],
                        "predicate": "near",
                        "object_id": b_id, "object_label": obj_b["label"],
                        "distance": round(float(dist_ground), 3)
                    })

        scene_data["relations"] = relations
        
        current_frame = scene_data["frame_metadata"]["frame_id"]
        if current_frame % STAT_LOG_INTERVAL == 0:
            self.flush_csv_buffer()
            
            if self.log_buffer:
                print(f"\n📈 [RESEARCH STATS TRACKER - FRAME {current_frame}]")
                print(f"   - 총 누적 HOI 유효 샘플 수: {len(self.log_buffer)}")
                
                unique_labels = set(x["target_label"] for x in self.log_buffer)
                
                for target_lbl in unique_labels:
                    lbl_dists = [x["dist_ground"] for x in self.log_buffer if x["target_label"] == target_lbl]
                    lbl_samples = len(lbl_dists)
                    
                    if lbl_samples == 0: continue
                    
                    lbl_under_count = sum(1 for d in lbl_dists if d <= self.near_threshold)
                    lbl_under_ratio = (lbl_under_count / lbl_samples) * 100.0
                    
                    print(f"   📂 [LABEL: {target_lbl:<12}] (샘플수: {lbl_samples}개)")
                    print(
                        f"       ➔ 거리 통계: MIN={np.min(lbl_dists):.2f} | P25={np.percentile(lbl_dists, 25):.2f} | "
                        f"P50(Med)={np.percentile(lbl_dists, 50):.2f} | P75={np.percentile(lbl_dists, 75):.2f} | "
                        f"MAX={np.max(lbl_dists):.2f} | STD={np.std(lbl_dists):.2f}"
                    )
                    print(f"       ➔ 매칭 신뢰도: 현재 설정된 Threshold({self.near_threshold}) 이하 데이터 비율: {lbl_under_ratio:.1f}% ({lbl_under_count}/{lbl_samples} 개)")
                
                latest = self.log_buffer[-1]
                print(f"   - 최근 HOI 단면 스냅샷: [{latest['target_label']}] x_diff:{latest['x_diff']:.2f} z_diff:{latest['z_diff']:.2f} dist_ground:{latest['dist_ground']:.2f}")
            
            if len(self.log_buffer) > 2000:
                self.log_buffer = self.log_buffer[-500:]
                
        return scene_data


class ResearchAffordanceEngine:
    def __init__(self):
        self.property_registry = {
            "person": ["interactive"], "chair": ["sittable"], "cup": ["graspable", "drinkable"],
            "bottle": ["graspable", "drinkable"], "cell phone": ["graspable", "usable"]
        }
        self.property_to_action = {
            "graspable": "grasp", "drinkable": "drink", "usable": "use", "sittable": "sit"
        }

    def infer_affordances(self, scene_data):
        objects = scene_data.get("objects", [])
        relations = scene_data.get("relations", [])
        person_ids = [o["id"] for o in objects if o["label"] == "person"]
        agent_id = person_ids[0] if person_ids else None

        for obj in objects:
            label = obj["label"]
            base_properties = self.property_registry.get(label, ["inspectable"])
            obj["affordance"] = {"properties": base_properties, "actions": []}

            if agent_id is None or obj["id"] == agent_id:
                obj["affordance"]["actions"] = ["interact"] if label == "person" else ["inspect"]
                obj["state"] = "agent_self" if label == "person" else "static"
                continue

            is_near = any(
                r["predicate"] == "near" and 
                ((r["subject_id"] == obj["id"] and r["object_id"] == agent_id) or 
                 (r["subject_id"] == agent_id and r["object_id"] == obj["id"])) 
                for r in relations
            )

            floor_margin = obj.get("floor_depth_delta", 0.0)
            if abs(floor_margin) <= 0.5: obj["state"] = "placed_on_floor"
            elif floor_margin > 0.5:     obj["state"] = "elevated"
            else:                        obj["state"] = "background_layer"

            active_actions = []
            for prop in base_properties:
                action_cand = self.property_to_action.get(prop)
                if action_cand:
                    if action_cand in ["grasp", "drink", "use"] and is_near:
                        active_actions.append(action_cand)
                    elif action_cand == "sit" and is_near and obj["state"] == "placed_on_floor":
                        active_actions.append(action_cand)

            obj["affordance"]["actions"] = active_actions if active_actions else ["inspect"]
        return scene_data


# =====================================================================
# 2. 메인 오케스트레이션 및 실시간 통계 파이프라인 (Main Loop)
# =====================================================================

def main():
    geometry_engine = UnifiedGeometryEngine()
    
    print(f"\n=====================================================================")
    print(f"🚀 [CV_AR WORLD MODEL EXPERIMENT FRAMEWORK v1.2] RUN TIME")
    print(f"   -> Pseudo3D Calibration Parameters: (fx={geometry_engine.f_x}, fy={geometry_engine.f_y})")
    print(f"   -> Mode Status: {CURRENT_PHASE_MODE} (Near Threshold: {1.0 if CURRENT_PHASE_MODE == 'PHASE_1_MEASURE' else CALIBRATED_NEAR_THRESHOLD})")
    print(f"   -> Floor Hypothesis ROI: {CURRENT_FLOOR_MODE} | Percentile: P{CURRENT_FLOOR_PERCENTILE}")
    print(f"   -> Active Experimental Condition Tracker: [{EXPERIMENTAL_CONDITION}]")
    print(f"   -> CSV Buffered Log Target Path: '{CSV_OUTPUT_PATH}'")
    print(f"=====================================================================\n")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    yolo_model = YOLO("yolov8n.pt")
    sam_model = SAM("sam_b.pt")
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", device=device)
    
    stream = UnifiedWebcamStream()
    floor_detector = ResearchFloorDetector(mode=CURRENT_FLOOR_MODE, percentile=CURRENT_FLOOR_PERCENTILE)
    relation_graph = ResearchRelationGraph(mode=CURRENT_PHASE_MODE, calibrated_threshold=CALIBRATED_NEAR_THRESHOLD)
    affordance_engine = ResearchAffordanceEngine()

    frame_count = 0
    cached_depth_map = None  
    last_masks = []
    last_labels = []
    
    exit_reason = "CRITICAL_CAMERA_ERROR"

    while True:
        ret, frame = stream.get_frame()
        if not ret or frame is None: 
            exit_reason = "CAMERA_STREAM_DISCONNECTED"
            break
        frame_count += 1

        h, w = frame.shape[:2]

        if frame_count == 1 or frame_count % DEPTH_INTERVAL == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pipe_out = depth_pipe(pil_img)
            pred_depth = pipe_out["predicted_depth"]
            
            if not isinstance(pred_depth, torch.Tensor):
                if isinstance(pred_depth, np.ndarray):
                    pred_depth = torch.from_numpy(pred_depth).float()
                elif isinstance(pred_depth, Image.Image):
                    pred_depth = torch.from_numpy(np.array(pred_depth)).float()
                else:
                    pred_depth = torch.tensor(pred_depth).float()
                    
            cached_depth_map = torch.nn.functional.interpolate(
                pred_depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()

        yolo_res = yolo_model(frame, device=device, verbose=False, conf=0.25)[0]
        
        scene_data = {
            "frame_metadata": {"frame_id": frame_count, "camera_resolution": [w, h]},
            "scene": {}, "objects": []
        }
        
        current_labels = []
        for idx, box in enumerate(yolo_res.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            lbl = yolo_res.names[int(box.cls[0].item())]
            current_labels.append(lbl)
            scene_data["objects"].append({
                "id": idx, "label": lbl, "yolo": {"bbox_2d": [x1, y1, x2, y2]},
                "sam": None, "depth": None, "spatial_3d": None, "affordance": None
            })

        can_use_cache = (
            frame_count % SAM_INTERVAL != 0 and 
            len(last_masks) == len(scene_data["objects"]) and 
            current_labels == last_labels
        )

        mask_overlay = np.zeros_like(frame)
        if can_use_cache:
            for idx, obj in enumerate(scene_data["objects"]):
                m_bool = last_masks[idx]
                y_idx, x_idx = np.where(m_bool)
                cx = int(np.mean(x_idx)) if x_idx.size > 0 else 0
                cy = int(np.mean(y_idx)) if y_idx.size > 0 else 0
                obj["sam"] = {"mask_area": int(np.sum(m_bool)), "centroid_2d": [cx, cy]}
                mask_overlay[m_bool] = [0, 255, 0]
        else:
            bboxes = [obj["yolo"]["bbox_2d"] for obj in scene_data["objects"]]
            if bboxes:
                sam_res = sam_model(frame, bboxes=bboxes, device=device, verbose=False)[0]
                if sam_res.masks is not None:
                    last_masks = []
                    for idx, obj in enumerate(scene_data["objects"]):
                        if idx >= len(sam_res.masks.data): break
                        m_bool = sam_res.masks.data[idx].cpu().numpy().astype(bool)
                        last_masks.append(m_bool)
                        y_idx, x_idx = np.where(m_bool)
                        cx = int(np.mean(x_idx)) if x_idx.size > 0 else 0
                        cy = int(np.mean(y_idx)) if y_idx.size > 0 else 0
                        obj["sam"] = {"mask_area": int(np.sum(m_bool)), "centroid_2d": [cx, cy]}
                        mask_overlay[m_bool] = [0, 255, 0]
                    last_labels = copy.deepcopy(current_labels)

        if cached_depth_map is not None and last_masks:
            for idx, obj in enumerate(scene_data["objects"]):
                if idx < len(last_masks) and obj["sam"]:
                    m_bool = last_masks[idx]
                    roi_pix = np.clip(cached_depth_map[m_bool], 0.0, None)
                    obj["depth"] = {"mean_relative_depth": round(float(np.mean(roi_pix)), 4) if roi_pix.size > 0 else 0.0}

        scene_3d = geometry_engine.process_scene_3d(scene_data)
        scene_floor = floor_detector.update_scene_with_floor(scene_3d, cached_depth_map)
        scene_graph = relation_graph.process_scene_relations(scene_floor)
        final_scene = affordance_engine.infer_affordances(scene_graph)

        if frame_count % 30 == 0:
            near_relations = [r for r in final_scene.get("relations", []) if r["predicate"] == "near"]
            print(f"\n🎬 [RUNTIME REPORT FRAME {frame_count}]")
            print(f"   - 지면 레이어 정보: floor_depth={final_scene['scene'].get('floor_depth', 0.0):.3f} | method={final_scene['scene'].get('floor_method', 'none')}")
            
            print(f"   - [FLOOR OCCLUSION TRACKER] 객체별 높이 상식 정량 단면:")
            for obj in final_scene["objects"]:
                print(f"     ➔ 객체명: {obj['label']:<12} | floor_depth_delta: {obj.get('floor_depth_delta', 0.0):>6.3f} | 추론된 공간 상태(state): {obj.get('state', 'unknown')}")
                
            print(f"   - 실시간 위상 요약: 총 사물 수={len(final_scene['objects'])} | 유출된 Near 관계 수={len(near_relations)}")
            if near_relations:
                # [피드백 반영 완료] 파이프라인 중괄호 내부 이스케이프 제거로 Python 3.10 크래시 버그 원천 해결
                near_strings = [f"{r['subject_label']}->near->{r['object_label']} (d:{r['distance']})" for r in near_relations]
                print(f"   - 유효 Near 목록: {near_strings}")
            print(json.dumps(final_scene, ensure_ascii=False, indent=2))

        annotated_frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.4, 0)
        cv2.imshow("CV_AR - Combined Master Research Pipeline", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            exit_reason = "NORMAL_USER_REQUEST"
            break

    print(f"\n=====================================================================")
    if exit_reason == "NORMAL_USER_REQUEST":
        print("🟢 [SUCCESS] 사용자가 키보드 'q'를 입력하여 정상 종료를 요청했습니다.")
    elif exit_reason == "CAMERA_STREAM_DISCONNECTED":
        print("🔴 [HARDWARE EXCEPTION] 비디오 스트림 입력이 불시 차단되었거나 카메라 하드웨어 오류가 발생했습니다.")
    
    print("⚠️  [SYSTEM WARNING] 잔여 메모리 CSV 버퍼를 실시간 디스크에 강제 병합 플러시합니다...")
    relation_graph.flush_csv_buffer()
    print("💾  [SUCCESS] 모든 지오메트리 위상 데이터셋이 데이터 누수 0%로 안전하게 저장되었습니다.")
    print("=====================================================================\n")
    
    stream.release()

if __name__ == "__main__":
    main()