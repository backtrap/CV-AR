import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.stream import CAMERA_MATRIX
from vision.reasoning.relation_graph import SpatialRelationGraph

def main():
    print("="*60)
    print("🧭 실시간 상하좌우 앞뒤(3D 위상) 관계 시각화 테스트")
    print("   -> YOLO + Depth (SAM 제외 초고속 모드)")
    print("   -> 두 개 이상의 사물을 비춰보세요!")
    print("="*60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("[시스템] 모델을 로드하는 중입니다...")
    yolo_model = YOLO("yolov8n.pt")
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", device=device)
    
    # 관계 엔진 로드 (near 임계값 0.7 고정)
    relation_graph = SpatialRelationGraph(near_threshold=0.7)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    f_x = float(CAMERA_MATRIX[0, 0])
    f_y = float(CAMERA_MATRIX[1, 1])
    c_x = float(CAMERA_MATRIX[0, 2])
    c_y = float(CAMERA_MATRIX[1, 2])
    
    # [스케일 보정] 사용자 측정치 반영 깊이 보정 계수 (실제 22cm / 화면 43cm = 약 0.51)
    DEPTH_SCALE_FACTOR = 0.51

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        
        # 1. Depth 추론
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pipe_out = depth_pipe(pil_img)
        pred_depth = pipe_out["predicted_depth"]
        if not isinstance(pred_depth, torch.Tensor):
            pred_depth = torch.from_numpy(np.array(pred_depth)).float()
        
        depth_map = torch.nn.functional.interpolate(
            pred_depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        
        # [핵심] 스케일 보정 적용
        depth_map = depth_map * DEPTH_SCALE_FACTOR

        # 2. YOLO 탐지
        yolo_res = yolo_model(frame, device=device, verbose=False, conf=0.3)[0]
        
        scene_data = {"objects": []}
        annotated_frame = frame.copy()

        # 3. 3D 좌표 변환 및 씬(scene) 데이터 구성
        for idx, box in enumerate(yolo_res.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            lbl = yolo_res.names[int(box.cls[0].item())]
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            roi_depth = depth_map[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
            z_val = float(np.mean(roi_depth)) if roi_depth.size > 0 else 0.0

            if z_val > 0:
                X = (cx - c_x) * z_val / f_x
                Y = (cy - c_y) * z_val / f_y
                
                obj_data = {
                    "id": idx, "label": lbl,
                    "spatial_3d": {"x": X, "y": Y, "z": z_val}
                }
                scene_data["objects"].append(obj_data)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"{lbl}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 4. 관계 그래프 연산 (상하좌우 앞뒤)
        relations = relation_graph.calculate_relations(scene_data)
        
        # 5. 관계 출력 (화면 좌측 상단)
        y_offset = 30
        cv2.putText(annotated_frame, "[Spatial Relations]", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30
        
        for rel in relations:
            subj = next((o["label"] for o in scene_data["objects"] if o["id"] == rel["subject_id"]), "Unknown")
            obj = next((o["label"] for o in scene_data["objects"] if o["id"] == rel["object_id"]), "Unknown")
            pred = rel["predicate"]
            
            # 읽기 쉬운 문장으로 포맷팅
            if pred == "left_of": text = f"- {subj} is LEFT of {obj}"
            elif pred == "right_of": text = f"- {subj} is RIGHT of {obj}"
            elif pred == "above": text = f"- {subj} is ABOVE {obj}"
            elif pred == "below": text = f"- {subj} is BELOW {obj}"
            elif pred == "closer_than": text = f"- {subj} is IN FRONT OF {obj}"
            elif pred == "farther_than": text = f"- {subj} is BEHIND {obj}"
            elif pred == "near": text = f"- {subj} is NEAR {obj}"
            else: text = f"- {subj} is {pred} {obj}"
            
            cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

        cv2.imshow("CV_AR Spatial Relation Test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
