import cv2
import json
import os
import numpy as np
import torch
from transformers import pipeline
from PIL import Image

# [Task 6] Depth 스케일 보정 계수를 코드 상수가 아니라 근거 파일에서 로드한다.
_DEPTH_SCALE_PATH = os.path.join(os.path.dirname(__file__), "depth_scale.json")


def load_depth_scale(path=_DEPTH_SCALE_PATH):
    """metric depth 선형 보정 파라미터(scale, offset)를 로드한다. true_m = scale*pred + offset.

    파일이 없거나 손상되면 과거 임시값(scale=0.51, offset=0.0)으로 안전 폴백한다.
    이 값들의 근거/이력은 depth_scale.json 의 provenance 필드에 남는다.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg.get("scale", 0.51)), float(cfg.get("offset", 0.0))
    except Exception as e:
        print(f"[Depth] depth_scale.json 로드 실패({e}) → 임시값 scale=0.51 사용")
        return 0.51, 0.0


# =====================================================================
# [Task 2 — 근거 있는 이상치 제거 상수]
# Depth-Anything-V2-Metric-Indoor 는 미터 단위 metric depth를 출력한다.
# 아래 값들은 "임의 garbage 필터"가 아니라 각각 명시적 근거를 갖는다.
# =====================================================================
MIN_VALID_DEPTH_M = 0.05   # 5cm 미만: 실내 팔길이보다 가까울 수 없음 → 모델/보정 노이즈로 간주
IQR_K = 1.5                # Tukey's fence 표준 계수(1.5×IQR). 경계값이 분포에서 도출됨
MIN_ROBUST_SAMPLES = 20    # IQR 추정에 필요한 최소 표본. 미만이면 fence 없이 median만 사용


def robust_representative_depth(roi_pixels):
    """SAM 마스크 영역의 depth 픽셀에서 이상치에 강건한 대표 깊이를 산출한다.

    방법(근거):
      1) 유효성 필터: 유한(finite)하고 MIN_VALID_DEPTH_M 이상인 픽셀만 사용.
         (0/음수/NaN은 metric depth로 성립 불가)
      2) Tukey 1.5×IQR 펜스로 이상치 제거.
         - 경계 임계값 [Q1-1.5·IQR, Q3+1.5·IQR]는 상수가 아니라 각 객체의
           분포(Q1,Q3)에서 자동 도출 → 데이터 기반 근거를 가진다.
         - 마스크 경계가 배경을 물어 생기는 원거리 tail을 제거한다.
      3) 대표값 = inlier의 median (평균과 달리 최대 50% 오염까지 강건).

    반환 dict:
      representative_depth : 강건 대표 깊이(m)
      mean_relative_depth  : 동일 값(다운스트림 하위호환 키)
      min/max_relative_depth : inlier 범위(이상치 제거 후)
      method, inlier_ratio, n_samples : 추적/검증용 메타데이터
    """
    zero = {
        "representative_depth": 0.0,
        "mean_relative_depth": 0.0,
        "min_relative_depth": 0.0,
        "max_relative_depth": 0.0,
        "method": "empty",
        "inlier_ratio": 0.0,
        "n_samples": 0,
    }
    if roi_pixels is None:
        return zero

    px = np.asarray(roi_pixels, dtype=np.float64).ravel()
    px = px[np.isfinite(px) & (px >= MIN_VALID_DEPTH_M)]
    n_total = int(px.size)
    if n_total == 0:
        return zero

    if n_total >= MIN_ROBUST_SAMPLES:
        q1, q3 = np.percentile(px, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        inliers = px[(px >= lower) & (px <= upper)]
        method = "iqr1.5_median"
        if inliers.size == 0:          # 분포가 극단적으로 좁아 전부 제외된 경우 방어
            inliers = px
            method = "iqr1.5_median_fallback"
    else:
        inliers = px
        method = "median_smallsample"

    rep = float(np.median(inliers))
    return {
        "representative_depth": round(rep, 4),
        "mean_relative_depth": round(rep, 4),
        "min_relative_depth": round(float(np.min(inliers)), 4),
        "max_relative_depth": round(float(np.max(inliers)), 4),
        "method": method,
        "inlier_ratio": round(float(inliers.size / n_total), 3),
        "n_samples": n_total,
    }


class DepthEstimator:
    def __init__(self):
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
        
        # 런타임 장치 로드 실패 시 CPU Fallback 예외 처리 적용
        try:
            print(f"Depth 모델을 [{self.device}] 장치에서 로드 시도 중...")
            self.pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
                device=self.device
            )
        except Exception as e:
            print(f"[경고] 장치 로드 실패: {e}. CPU(-1)로 대체합니다.")
            self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", device=-1)
        # [Task 6] depth 선형 보정 계수 로드 (근거: depth_scale.json)
        self.depth_scale, self.depth_offset = load_depth_scale()
        print(f"[Depth] 보정 계수 로드: true_m = {self.depth_scale}*pred + {self.depth_offset}")
        print("Depth Anything V2 Metric 모델이 성공적으로 준비되었습니다.")

    def get_depth_map(self, frame):
        """
        입력 프레임의 해상도를 동적으로 감지하여 시각화용(0-255)과 정밀 계산용(float) 맵을 분리 반환합니다.
        """
        h, w = frame.shape[:2]
        
        # 1. OpenCV BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # [TypeError 해결] NumPy 배열을 Hugging Face 파이프라인이 요구하는 PIL Image 객체로 정밀 래핑
        pil_image = Image.fromarray(rgb_frame)
        
        # 2. 깊이 추론 실행
        pipe_output = self.pipe(pil_image)
        predicted_depth = pipe_output["predicted_depth"]
        
        # 데이터 타입 검증 (Tensor로 강제 통일 및 보간 준비)
        if not isinstance(predicted_depth, torch.Tensor):
            if isinstance(predicted_depth, Image.Image):
                predicted_depth = torch.from_numpy(np.array(predicted_depth)).float()
        
        # 3. 원본 해상도로 정밀 보간 (Raw Float Depth Map 확보 - 5단계 연산용)
        raw_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

        # [스케일 보정] 선형 보정 true_m = scale*pred + offset (계수 근거: depth_scale.json)
        raw_depth = raw_depth * self.depth_scale + self.depth_offset

        return {"raw_depth": raw_depth}
             
        ##### visual depth 연산은 메인로직에서 불필요함으로 제거
        
        # # 4. 시각화용 0~255 정규화 맵 생성
        # d_min, d_max = raw_depth.min(), raw_depth.max()
        # if d_max - d_min > 1e-5:
        #     visual_depth = (raw_depth - d_min) / (d_max - d_min) * 255.0
        # else:
        #     visual_depth = np.zeros_like(raw_depth)

        # return {
        #     "visual_depth": visual_depth.astype(np.uint8),
        #     "raw_depth": raw_depth
        # }

    def get_object_depth(self, depth_dict, mask_bool):
        """
        SAM 마스크(True/False 행렬) 영역만 슬라이싱하여 오염 없는 순수 객체 깊이 통계를 추출합니다.
        """
        if mask_bool is None or np.sum(mask_bool) == 0:
            return {"mean_relative_depth": 0.0, "min_relative_depth": 0.0, "max_relative_depth": 0.0}

        # 정밀 계산용 raw_depth 행렬에서 마스크 영역 픽셀만 추출
        roi_pixels = depth_dict["raw_depth"][mask_bool]

        return {
            "mean_relative_depth": round(float(np.mean(roi_pixels)), 4),
            "min_relative_depth": round(float(np.min(roi_pixels)), 4),
            "max_relative_depth": round(float(np.max(roi_pixels)), 4)
        }

# =====================================================================
# 🚀 [4단계 단독 모듈 구동 및 마그마 컬러 가시화 테스트 베드]
# =====================================================================
if __name__ == "__main__":
    from vision.stream import WebcamStream
    stream = WebcamStream()
    estimator = DepthEstimator()
    
    print("==========================================================")
    print("CV_AR: 4단계 Depth 단독 검증을 시작합니다.")
    print("-> 가까운 물체는 밝고 환하게(노란색/흰색), 먼 배경은 어둡게(보라색/검은색) 출력됩니다.")
    print("-> 'q'를 누르면 안전하게 종료됩니다.")
    print("==========================================================")
    
    while True:
        ret, frame = stream.get_frame()
        if not ret: 
            break
        
        # 실시간 깊이 지도 데이터 딕셔너리 획득
        depth_data = estimator.get_depth_map(frame)
        
        # 의사 컬러맵(Magma)을 입혀 시각적 직관성 확보
        color_depth = cv2.applyColorMap(depth_data["visual_depth"], cv2.COLORMAP_MAGMA)
        
        # 화면 팝업 출력
        cv2.imshow("CV_AR - Depth Anything V2 (Validated)", color_depth)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
    stream.release()
    cv2.destroyAllWindows()