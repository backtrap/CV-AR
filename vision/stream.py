import cv2
import numpy as np

CAMERA_MATRIX = np.array([
    [759.75630166, 0.0, 635.88491447],
    [0.0, 759.86158408, 370.11350743],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([
    1.79291483e-01,
    -4.60703568e-01,
    -2.48965512e-04,
    1.31282385e-03,
    3.42945994e-01
], dtype=np.float32)

class WebcamStream:
    """
    CV_AR 프로젝트의 실시간 카메라 스트리밍 및 영상 공급을 전담하는 클래스입니다.
    추후 YOLO, SAM, Depth Anything V2 모듈이 이 클래스로부터 프레임을 공급받습니다.
    """
    def __init__(self, video_source=0):
        # 0번: 맥북 기본 내장 FaceTime HD 카메라 인식
        self.cap = cv2.VideoCapture(video_source)
        
        # M4 맥북의 성능과 비전 연산 속도를 고려한 최적 해상도(720p HD) 고정
        
        # 1050 사용 이슈로 임시로 해상도 낮춤, 카메라 임시 행렬도 조정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # -----------------------------------------------------------------
        # [CRITICAL WARNING] 카메라 행렬(Intrinsic Matrix) 설정
        # 현재 값은 1280x720 해상도에 맞춘 임시 추정치(900, 640, 360)입니다.
        # 1~5단계(2D 객체 및 깊이 탐지)까지는 이 임시값으로도 완벽히 구동되나,
        # 6단계(2D -> 3D 좌표 변환) 진입 직전에는 반드시 OpenCV 체커보드 
        # 캘리브레이션을 수행하여 실제 맥북 M4 카메라의 고유 서명 값으로 교체해야
        # 아바타 상호작용의 물리적 오차(수십 cm 단위)를 방지할 수 있습니다.
        # -----------------------------------------------------------------
        self.camera_matrix = CAMERA_MATRIX.copy()
        self.dist_coeffs = DIST_COEFFS.copy()
        
    def get_frame(self):
        """
        카메라로부터 현재 프레임을 읽어와 안전하게 복사본을 반환합니다.
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        
        # 원본 프레임 오염 방지를 위해 .copy()하여 반환
        return ret, frame.copy()

    def release(self):
        """
        카메라 자원을 해제하고 열려있는 모든 OpenCV 윈도우 창을 닫습니다.
        """
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 1. OpenCV 실시간 웹캠 스트림 구축 단독 검증 테스트
    # -----------------------------------------------------------------
    stream = WebcamStream()
    print("==========================================================")
    print("CV_AR: 1번 마일스톤 [OpenCV 실시간 웹캠 스트림] 검증을 시작합니다.")
    print("-> 카메라 화면 창을 클릭한 후 'q'를 누르면 안전하게 종료됩니다.")
    print("==========================================================")
    
    while True:
        ret, frame = stream.get_frame()
        if not ret:
            print("[에러] 맥북 카메라로부터 영상을 읽어올 수 없습니다.")
            break
            
        # 화면에 실시간 스트림 출력
        cv2.imshow("CV_AR - Webcam Base Stream", frame)
        
        # 키보드 'q' 입력 감지 시 루프 탈출 (1ms 대기)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자 요청으로 스트림을 안전하게 종료합니다.")
            break
            
    # 자원 해제
    stream.release()
