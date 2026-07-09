import cv2
import numpy as np
import time
import json
import copy
import torch
import csv
import os
import glob
import re
from vision.spatial.floor_detector import FloorPlaneDetector
from ultralytics import YOLO, SAM
from transformers import pipeline
from PIL import Image

# =====================================================================
# 🔬 [최종 진화형] Floor 검증 하네스 - 데이터 채굴 마스터 스크립트
# =====================================================================
CALIBRATED_NEAR_THRESHOLD = 0.6
CURRENT_FLOOR_MODE = "Bottom_ROI"
SAM_INTERVAL = 5
DEPTH_INTERVAL = 10
STAT_LOG_INTERVAL = 60

def generate_next_filename():
    target_dir = "data"
    os.makedirs(target_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(target_dir, "floor_validation_log*.csv"))
    max_idx = 0
    pattern = re.compile(r"floor_validation_log(\d+)\.csv")
    for f in existing_files:
        match = pattern.search(os.path.basename(f))
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return os.path.join(target_dir, f"floor_validation_log{max_idx + 1:02d}.csv")

CSV_OUTPUT_PATH = generate_next_filename()


class UnifiedWebcamStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame.copy() if ret else None
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

class UnifiedGeometryEngine:
    def __init__(self, f_x=900.0, f_y=900.0):
        self.f_x = f_x
        self.f_y = f_y
    def convert_to_3d(self, centroid_2d, mean_relative_depth, c_x, c_y):
        u, v = centroid_2d
        z_val = max(mean_relative_depth, 0.001)
        X = (u - c_x) * z_val / self.f_x
        Y = (v - c_y) * z_val / self.f_y
        return {"x": round(float(X), 3), "y": round(float(Y), 3), "z": round(float(z_val), 3)}
    def process_scene_3d(self, scene_data):
        resolution = scene_data["frame_metadata"]["camera_resolution"]
        c_x, c_y = resolution[0] / 2.0, resolution[1] / 2.0
        for obj in scene_data["objects"]:
            if obj.get("sam") and obj.get("depth"):
                obj["spatial_3d"] = self.convert_to_3d(obj["sam"]["centroid_2d"], obj["depth"]["mean_relative_depth"], c_x, c_y)
        return scene_data

class ResearchFloorDetector:
    def __init__(self, roi_ratio=0.25):
        self.roi_ratio = roi_ratio
    def detect_floor(self, depth_map):
        if depth_map is None: return 0.0, 0
        h = depth_map.shape[0]
        roi_start_y = int(h * (1.0 - self.roi_ratio))
        floor_roi = depth_map[roi_start_y:h, :]
        valid = floor_roi[floor_roi > 0]
        return (float(np.median(valid)), int(valid.size)) if valid.size > 0 else (0.0, 0)

    def update_scene_with_floor(self, scene_data, raw_depth):
        floor_depth, valid_pixels = self.detect_floor(raw_depth)
        scene_data["scene"]["floor_depth"] = round(floor_depth, 3)
        scene_data["scene"]["floor_valid_pixels"] = valid_pixels
        scene_data["scene"]["floor_method"] = "bottom_roi_median(P50)"
        for obj in scene_data["objects"]:
            if obj.get("spatial_3d"):
                obj_z = obj["spatial_3d"]["z"]
                obj["floor_depth_delta"] = round(float(floor_depth - obj_z), 3) if obj_z > 0 else 0.0
        return scene_data

class ResearchRelationGraph:
    def __init__(self, threshold=3.0):
        self.near_threshold = threshold
        self.log_buffer = []
        self.csv_buffer = []

        os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
        if not os.path.exists(CSV_OUTPUT_PATH):
            with open(CSV_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "label", "floor_depth", "target_z", "floor_depth_delta", "near_distance", "mask_area", "centroid_y", "floor_valid_pixels"])

    def flush_csv(self):
        if self.csv_buffer:
            with open(CSV_OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.csv_buffer)
            self.csv_buffer.clear()

    def process_scene_relations(self, scene_data):
        relations = []
        objects = scene_data.get("objects", [])
        num_objs = len(objects)
        floor_depth = scene_data["scene"]["floor_depth"]
        valid_pixels = scene_data["scene"].get("floor_valid_pixels", 0)
        frame_id = scene_data["frame_metadata"]["frame_id"]

        person_objs = [o for o in objects if o["label"] == "person" and o.get("sam")]
        agent_obj = max(person_objs, key=lambda x: x["sam"]["mask_area"]) if person_objs else None
        agent_id = agent_obj["id"] if agent_obj else None

        for i in range(num_objs):
            for j in range(i + 1, num_objs):
                obj_a, obj_b = objects[i], objects[j]
                if not obj_a.get("spatial_3d") or not obj_b.get("spatial_3d"): continue

                x_diff = obj_a["spatial_3d"]["x"] - obj_b["spatial_3d"]["x"]
                z_diff = obj_a["spatial_3d"]["z"] - obj_b["spatial_3d"]["z"]
                dist_ground = round(float(np.sqrt(x_diff**2 + z_diff**2)), 4)

                if dist_ground <= self.near_threshold:
                    relations.append({
                        "subject_id": obj_a["id"], "subject_label": obj_a["label"],
                        "predicate": "near", "object_id": obj_b["id"], "object_label": obj_b["label"], "distance": round(dist_ground, 3)
                    })

                is_hoi = (agent_id is not None) and ((obj_a["id"] == agent_id and obj_b["label"] != "person") or (obj_b["id"] == agent_id and obj_a["label"] != "person"))
                if is_hoi:
                    target_obj = obj_b if obj_a["id"] == agent_id else obj_a
                    delta = target_obj.get("floor_depth_delta", 0.0)
                    m_area = target_obj["sam"]["mask_area"] if target_obj.get("sam") else 0
                    c_y = target_obj["sam"]["centroid_2d"][1] if target_obj.get("sam") else 0

                    entry = {"label": target_obj["label"], "delta": delta, "target_z": target_obj["spatial_3d"]["z"]}
                    self.log_buffer.append(entry)

                    self.csv_buffer.append([frame_id, target_obj["label"], floor_depth, target_obj["spatial_3d"]["z"], delta, dist_ground, m_area, c_y, valid_pixels])

        scene_data["relations"] = relations

        if frame_id % STAT_LOG_INTERVAL == 0:
            self.flush_csv()
            if self.log_buffer:
                print(f"\n📊 [FLOOR LAYER VALIDATION REPORT - FRAME {frame_id}]")
                print(f"   - 총 누적 측정 샘플 수: {len(self.log_buffer)}개 | 실시간 ROI 유효 픽셀 수: {valid_pixels} px")
                labels = set(x["label"] for x in self.log_buffer)

                for lbl in labels:
                    sub_samples = [x for x in self.log_buffer if x["label"] == lbl]
                    deltas = [x["delta"] for x in sub_samples]
                    mean_z = np.mean([x["target_z"] for x in sub_samples])
                    if not deltas: continue
                    print(f"   📂 [객체 라벨: {lbl:<12}] (평균 target_z: {mean_z:.2f}m)")
                    print(
                        f"       ➔ floor_depth_delta 통계: MIN={np.min(deltas):.2f} | "
                        f"P25={np.percentile(deltas, 25):.2f} | P50(Med)={np.percentile(deltas, 50):.2f} | "
                        f"P75={np.percentile(deltas, 75):.2f} | MAX={np.max(deltas):.2f} | STD={np.std(deltas):.2f}"
                    )
            if len(self.log_buffer) > 1500:
                self.log_buffer = self.log_buffer[-300:]

        return scene_data

class ResearchAffordanceEngine:
    def __init__(self):
        self.property_registry = {
            "person": ["interactive"], "cup": ["graspable", "drinkable"],
            "bottle": ["graspable", "drinkable"], "wine glass": ["graspable", "drinkable"],
            "cell phone": ["graspable", "usable"], "remote": ["graspable", "usable"]
        }
        self.property_to_action = {"graspable": "grasp", "drinkable": "drink", "usable": "use"}

    def infer_affordances(self, scene_data):
        objects = scene_data.get("objects", [])
        relations = scene_data.get("relations", [])

        person_objs = [o for o in objects if o["label"] == "person" and o.get("sam")]
        agent_id = max(person_objs, key=lambda x: x["sam"]["mask_area"])["id"] if person_objs else None

        for obj in objects:
            label = obj["label"]
            base_properties = self.property_registry.get(label, ["inspectable"])
            obj["affordance"] = {"properties": base_properties, "actions": []}

            if agent_id is None or obj["id"] == agent_id:
                obj["affordance"]["actions"] = ["interact"] if label == "person" else ["inspect"]
                continue

            is_near = any(
                r["predicate"] == "near" and
                ((r["subject_id"] == obj["id"] and r["object_id"] == agent_id) or
                 (r["subject_id"] == agent_id and r["object_id"] == obj["id"]))
                for r in relations
            )

            active_actions = []
            for prop in base_properties:
                action_cand = self.property_to_action.get(prop)
                if action_cand and is_near: active_actions.append(action_cand)
            obj["affordance"]["actions"] = active_actions if active_actions else ["inspect"]
        return scene_data

def main():
    print(f"\n=====================================================================")
    print(f"🔬 [CV_AR WORLD MODEL VALIDATION HARNESS v1.2] - FINAL TUNED")
    print(f"   -> CSV 저장 경로: {CSV_OUTPUT_PATH}")
    print(f"=====================================================================\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    yolo_model = YOLO("yolov8n.pt")
    sam_model = SAM("sam_b.pt")
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", device=device)

    stream = UnifiedWebcamStream()
    geometry_engine = UnifiedGeometryEngine()
    floor_detector = FloorPlaneDetector()
    relation_graph = ResearchRelationGraph(threshold=CALIBRATED_NEAR_THRESHOLD)
    affordance_engine = ResearchAffordanceEngine()

    frame_count = 0
    cached_depth_map = None
    last_masks, last_labels = [], []

    while True:
        ret, frame = stream.get_frame()
        if not ret or frame is None: break
        frame_count += 1
        h, w = frame.shape[:2]

        if frame_count == 1 or frame_count % DEPTH_INTERVAL == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pipe_out = depth_pipe(pil_img)
            pred_depth = pipe_out["predicted_depth"]
            if not isinstance(pred_depth, torch.Tensor):
                pred_depth = torch.from_numpy(np.array(pred_depth)).float()
            cached_depth_map = torch.nn.functional.interpolate(
                pred_depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
            
            # [스케일 보정] 사용자 측정치 반영 깊이 보정 계수 추가 (실제 22cm / 화면 43cm = 약 0.51)
            DEPTH_SCALE_FACTOR = 0.51
            cached_depth_map = cached_depth_map * DEPTH_SCALE_FACTOR

        yolo_res = yolo_model(frame, device=device, verbose=False, conf=0.25)[0]
        scene_data = {"frame_metadata": {"frame_id": frame_count, "camera_resolution": [w, h]}, "scene": {}, "objects": []}

        current_labels = []
        for idx, box in enumerate(yolo_res.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            lbl = yolo_res.names[int(box.cls[0].item())]
            current_labels.append(lbl)
            scene_data["objects"].append({"id": idx, "label": lbl, "yolo": {"bbox_2d": [x1, y1, x2, y2]}, "sam": None, "depth": None})

        can_use_cache = (frame_count % SAM_INTERVAL != 0 and len(last_masks) == len(scene_data["objects"]) and current_labels == last_labels)
        if can_use_cache:
            for idx, obj in enumerate(scene_data["objects"]):
                m_bool = last_masks[idx]
                cx = int(np.mean(np.where(m_bool)[1])) if np.any(m_bool) else 0
                cy = int(np.mean(np.where(m_bool)[0])) if np.any(m_bool) else 0
                obj["sam"] = {"mask_area": int(np.sum(m_bool)), "centroid_2d": [cx, cy]}
        else:
            bboxes = [obj["yolo"]["bbox_2d"] for obj in scene_data["objects"]]
            if bboxes:
                sam_res = sam_model(frame, bboxes=bboxes, device=device, verbose=False)[0]
                if sam_res.masks is not None:
                    last_masks = [m.cpu().numpy().astype(bool) for m in sam_res.masks.data]
                    for idx, obj in enumerate(scene_data["objects"]):
                        if idx >= len(last_masks): break
                        m_bool = last_masks[idx]
                        cx = int(np.mean(np.where(m_bool)[1])) if np.any(m_bool) else 0
                        cy = int(np.mean(np.where(m_bool)[0])) if np.any(m_bool) else 0
                        obj["sam"] = {"mask_area": int(np.sum(m_bool)), "centroid_2d": [cx, cy]}
                    last_labels = copy.deepcopy(current_labels)

        if cached_depth_map is not None and last_masks:
            for idx, obj in enumerate(scene_data["objects"]):
                if idx < len(last_masks) and obj["sam"]:
                    roi_pix = np.clip(cached_depth_map[last_masks[idx]], 0.0, None)
                    obj["depth"] = {"mean_relative_depth": round(float(np.mean(roi_pix)), 4) if roi_pix.size > 0 else 0.0}

        scene_3d = geometry_engine.process_scene_3d(scene_data)
        scene_floor = floor_detector.update_scene_with_floor(scene_3d, {"raw_depth": cached_depth_map})
        scene_graph = relation_graph.process_scene_relations(scene_floor)
        final_scene = affordance_engine.infer_affordances(scene_graph)

        if frame_count % 30 == 0:
            print(f"\n🎬 [RUN FREQUENCY FRAGMENT - {frame_count}]")
            print(f"   [Baseline Floor P50]: {final_scene['scene'].get('floor_depth')} | Valid Pixels: {final_scene['scene'].get('floor_valid_pixels')}")
            for obj in final_scene["objects"]:
                if obj["label"] != "person" and obj.get("spatial_3d"):
                    print(f"     ➔ {obj['label']:<12}: target_z={obj['spatial_3d']['z']:.3f} | floor_depth_delta={obj.get('floor_depth_delta'):.3f} | Actions={obj['affordance']['actions']}")
            near_rels = [r for r in final_scene.get("relations", []) if r["predicate"] == "near"]
            if near_rels:
                near_strings = [f"{r['subject_label']}↔{r['object_label']}(d:{r['distance']})" for r in near_rels]
                print(f"   [Active Near Connections]: {near_strings}")

        cv2.imshow("CV_AR - World Model Validation Harness", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("\n⚠️ 디스크 플러시 및 세션 종료 중...")
    relation_graph.flush_csv()
    print(f"💾 [SUCCESS] 모든 로그 데이터셋이 '{CSV_OUTPUT_PATH}'에 안전하게 저장되었습니다.\n")
    stream.release()

if __name__ == "__main__":
    main()