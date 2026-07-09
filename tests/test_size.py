import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

import sys
import os
import csv
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.stream import CAMERA_MATRIX

def main():
    print("="*60)
    print("📏 실시간 사물 크기(cm) 측정 테스트")
    print("   -> YOLO 탐지 + Depth Anything V2 + 캘리브레이션 투영")
    print("   -> 'q' 키를 누르면 종료됩니다.")
    print("="*60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("[시스템] 모델을 로드하는 중입니다...")
    yolo_model = YOLO("yolov8n.pt")
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", device=device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 캘리브레이션 행렬에서 초점거리(focal length) 추출
    f_x = float(CAMERA_MATRIX[0, 0])
    f_y = float(CAMERA_MATRIX[1, 1])
    c_x = float(CAMERA_MATRIX[0, 2])
    c_y = float(CAMERA_MATRIX[1, 2])

    # CSV 로깅 파일 초기화
    os.makedirs("data", exist_ok=True)
    csv_filename = f"data/size_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "label", "z_val_m", "real_w_cm", "real_h_cm", "TL_X", "TL_Y", "TR_X", "TR_Y", "BL_X", "BL_Y", "BR_X", "BR_Y"])

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("[에러] 카메라 프레임을 읽어올 수 없습니다.")
            break

        h, w = frame.shape[:2]
        
        # 1. Depth 측정 (Metric 깊이 추론)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pipe_out = depth_pipe(pil_img)
        pred_depth = pipe_out["predicted_depth"]
        if not isinstance(pred_depth, torch.Tensor):
            pred_depth = torch.from_numpy(np.array(pred_depth)).float()
        
        depth_map = torch.nn.functional.interpolate(
            pred_depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        # 2. YOLO 탐지
        yolo_res = yolo_model(frame, device=device, verbose=False, conf=0.3)[0]

        annotated_frame = frame.copy()

        # 3. 크기 계산 및 시각화
        for box in yolo_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            lbl = yolo_res.names[int(box.cls[0].item())]
            
            # 박스 중심점의 Z값 추출 (노이즈 방지를 위해 박스 중앙 10x10 픽셀 영역의 평균값 사용)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            roi_depth = depth_map[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
            
            # [스케일 보정] 사용자 측정치 반영 (실제 22cm / 화면 43cm = 약 0.51)
            DEPTH_SCALE_FACTOR = 0.72
            z_val = float(np.mean(roi_depth)) if roi_depth.size > 0 else 0.0
            z_val = z_val * DEPTH_SCALE_FACTOR


            if z_val > 0:
                pixel_w = abs(x2 - x1)
                pixel_h = abs(y2 - y1)
                
                # 핀홀 카메라 모델 역투영 공식을 이용한 물리 크기 환산 (m -> cm 변환을 위해 100 곱하기)
                real_w_cm = (pixel_w * z_val / f_x) * 100.0
                real_h_cm = (pixel_h * z_val / f_y) * 100.0
                
                # PPS(Peripersonal Space) 70cm(0.7m) 기준 보정 및 시각화
                if z_val <= 0.7:
                    box_color = (0, 0, 255)  # BGR: Red (PPS 진입 시)
                    text = f"[PPS] {lbl} ({real_w_cm:.1f}x{real_h_cm:.1f}cm) | D: {z_val:.2f}m"
                else:
                    box_color = (0, 255, 0)  # BGR: Green (PPS 밖)
                    text = f"{lbl} ({real_w_cm:.1f}x{real_h_cm:.1f}cm) | D: {z_val:.2f}m"
                
                # 박스 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # 텍스트 오버레이
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                
                # 아바타 기준 3D 현실 공간 꼭짓점 좌표 (미터 단위)
                TL_X = (x1 - c_x) * z_val / f_x
                TL_Y = (y1 - c_y) * z_val / f_y
                TR_X = (x2 - c_x) * z_val / f_x
                TR_Y = (y1 - c_y) * z_val / f_y
                BL_X = (x1 - c_x) * z_val / f_x
                BL_Y = (y2 - c_y) * z_val / f_y
                BR_X = (x2 - c_x) * z_val / f_x
                BR_Y = (y2 - c_y) * z_val / f_y

                # 3D 좌표 오버레이 (박스 아래쪽)
                coord_text1 = f"TL:({TL_X:.2f}, {TL_Y:.2f}) TR:({TR_X:.2f}, {TR_Y:.2f})"
                coord_text2 = f"BL:({BL_X:.2f}, {BL_Y:.2f}) BR:({BR_X:.2f}, {BR_Y:.2f})"
                cv2.putText(annotated_frame, coord_text1, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated_frame, coord_text2, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # CSV 파일에 현재 프레임의 결과값 기록
                csv_writer.writerow([
                    datetime.now().strftime('%H:%M:%S.%f')[:-3],
                    lbl, round(z_val, 4), round(real_w_cm, 2), round(real_h_cm, 2),
                    round(TL_X, 4), round(TL_Y, 4), round(TR_X, 4), round(TR_Y, 4),
                    round(BL_X, 4), round(BL_Y, 4), round(BR_X, 4), round(BR_Y, 4)
                ])

        # 화면 출력
        cv2.imshow("CV_AR Size Measurement Test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # 종료 시 CSV 파일 저장 및 닫기
    csv_file.close()
    print(f"\n💾 실시간 크기 측정 및 3D 좌표 로그가 '{csv_filename}'에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    main()
