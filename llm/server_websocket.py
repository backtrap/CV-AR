"""
유니티 연동을 위한 파이썬 WebSocket 서버
실시간 웹캠 -> 비전 파이프라인 -> Gemini (10초 쿨다운) -> WebSocket 브로드캐스트
"""

import asyncio
import websockets
import json
import time
import cv2
import threading
import queue
from PIL import Image

from vision.stream import WebcamStream, CAMERA_MATRIX
from vision.detector import ObjectDetector
from vision.segmentation.segmenter import ObjectSegmenter, SceneDepthAttacher
from vision.depth.depth_estimator import DepthEstimator
from vision.spatial.transformer import Spatial3DConverter
from vision.spatial.stabilizer import CoordinateStabilizer
from vision.reasoning.relation_graph import SpatialRelationGraph
from vision.reasoning.affordance_engine import AffordanceEngine
from vision.spatial.floor_detector import FloorPlaneDetector

from llm.feature_extractor import build_inputs_from_scene, DEFAULT_CONTEXT
from llm.interpreter import interpret_batch
from llm.schemas import SemanticInterpretationBatchInput, SemanticInterpretationBatchOutput, SemanticInterpretationOutput

# [TEST MODE] API 한도 회피용 모의(Mock) LLM 활성화 플래그
MOCK_LLM = False

# SAM은 계산 비용이 크므로 일정 주기 (5프레임)마다 새로 실행하고, 그 사이에는 이전 마스크를 재사용합니다.
# 단, 객체의 위치가 크게 변한 경우에는 주기 전이라도 즉시 SAM을 다시 실행합니다.
SAM_INTERVAL = 6
SAM_IOU_THRESHOLD = 0.7

# 글로벌 변수
connected_clients = set()

# VLM 요청은 최신 장면 하나만 보관합니다.
# 추론이 느린 동안 과거 프레임 요청이 누적되는 것을 막기 위해 큐 크기를 1로 고정합니다.
vlm_queue = queue.Queue(maxsize=1)

# 비전 파이프라인 모듈들
detector = None
segmenter = None
depth_estimator = None
depth_attacher = None
spatial_converter = None
stabilizer = None
relation_graph = None
affordance_engine = None
floor_detector = None

def init_vision_modules():
    global detector, segmenter, depth_estimator, depth_attacher, spatial_converter, stabilizer, relation_graph, affordance_engine, floor_detector
    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth_estimator = DepthEstimator()
    depth_attacher = SceneDepthAttacher()
    spatial_converter = Spatial3DConverter(
        camera_matrix=CAMERA_MATRIX
    )
    stabilizer = CoordinateStabilizer()
    relation_graph = SpatialRelationGraph()
    affordance_engine = AffordanceEngine()
    floor_detector = FloorPlaneDetector()

    print("[서버] 비전 모듈 초기화 완료.")

def bbox_iou(box_a, box_b) -> float:
    """
    [x1, y1, x2, y2] 형식인 두 YOLO bbox의 IoU를 계산합니다.
    객체 박스 간 겹침 정도를 파악하는데 사용합니다.
    """
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area

    return intersection_area / union_area if union_area > 0.0 else 0.0


def build_scene_graph_for_frame(
    frame,
    frame_count: int,
    sam_frame_count: int,
    sam_cache: dict,
) -> dict:
    """YOLO는 매번 실행하고, 조건이 안전할 때만 이전 SAM 마스크를 재사용합니다."""
    # 객체 목록과 bbox는 프레임마다 달라질 수 있으므로 YOLO 처리 과정은 그대로 유지합니다.
    result = detector.detect(frame)
    scene_data = detector.build_scene(result, frame, frame_count)

    current_objects = scene_data.get("objects", [])
    current_labels = [obj["label"] for obj in current_objects]
    current_bboxes = [obj["yolo"]["bbox_2d"] for obj in current_objects]

    # 라벨과 객체 수가 같아도 bbox가 크게 이동했다면 과거 마스크는 더 이상 유효하지 않습니다.
    # 모든 객체의 IoU가 기준값 이상일 때만 캐시를 안전하게 재사용합니다.
    bboxes_are_stable = (
        len(current_bboxes) == len(sam_cache["last_bboxes"])
        and all(
            bbox_iou(previous_bbox, current_bbox) >= SAM_IOU_THRESHOLD
            for previous_bbox, current_bbox in zip(
                sam_cache["last_bboxes"], current_bboxes
            )
        )
    )
    can_reuse_masks = (
        bool(sam_cache["last_masks_list"])
        and bool(sam_cache["last_sam_data"])
        and sam_cache["cached_depth_map"] is not None
        and all(sam_data is not None for sam_data in sam_cache["last_sam_data"])
        and sam_frame_count % SAM_INTERVAL != 0
        and len(current_objects) == len(sam_cache["last_masks_list"])
        and len(current_objects) == len(sam_cache["last_sam_data"])
        and current_labels == sam_cache["last_labels"]
        and bboxes_are_stable
    )

    if can_reuse_masks:
        # SAM과 Depth를 한 세트로 재사용합니다. 이 경로에서는 두 AI 모델 모두 실행하지 않습니다.
        _, scene_data = segmenter.overlay_cached_masks(
            frame, scene_data, sam_cache["last_sam_data"]
        )
        masks_list = sam_cache["last_masks_list"]
        depth_map = sam_cache["cached_depth_map"]
    else:
        # SAM 갱신이 필요하면 같은 프레임으로 Depth도 함께 계산하여 두 캐시의 시점을 맞춥니다.
        _, scene_data, masks_list = segmenter.segment_objects(frame, scene_data)
        depth_map = depth_estimator.get_depth_map(frame)

        sam_cache["last_masks_list"] = masks_list
        sam_cache["last_sam_data"] = [
            obj["sam"].copy() if obj.get("sam") is not None else None
            for obj in scene_data["objects"]
        ]
        sam_cache["cached_depth_map"] = depth_map

    # 다음 처리 프레임의 캐시 유효성 판단을 위해 최신 YOLO 결과를 보관합니다.
    sam_cache["last_labels"] = current_labels
    sam_cache["last_bboxes"] = current_bboxes

    # 새로 계산했거나 캐시에서 가져온 동일 시점의 SAM 마스크와 Depth map을 결합합니다.
    scene_data = depth_attacher.attach_depth(scene_data, masks_list, depth_map)
    scene_data = spatial_converter.process_scene_3d(scene_data)
    # [Task4] 3D 변환 직후 좌표 안정화(1€ 필터)로 프레임 간 지터 제거
    scene_data = stabilizer.process_scene(scene_data)
    # <--- 추가: 3D 변환 직후에 바닥을 감지하여 델타값을 구합니다 --->
    scene_data = floor_detector.update_scene_with_floor(scene_data, depth_map)
    scene_data = relation_graph.process_scene_relations(scene_data)
    # <--- 추가: 관계 그래프 완성 직후 행동 추론 엔진을 통과시킵니다 --->
    scene_data = affordance_engine.infer_affordances(scene_data)
    return scene_data

def is_significant_change(prev_inputs, curr_inputs):
    """
    delta tirgger에 관한 함수    
    """
    if prev_inputs is None:
        return True
    if len(prev_inputs) != len(curr_inputs):
        return True
    for p_obj, c_obj in zip(prev_inputs, curr_inputs):
        if abs(p_obj.mask_area - c_obj.mask_area) / max(p_obj.mask_area, 1) > 0.5:
            return True
        previous_box = p_obj.bbox_2d or []
        current_box = c_obj.bbox_2d or []
        if bbox_iou(previous_box, current_box) < SAM_IOU_THRESHOLD:
            return True
        if abs(p_obj.target_z - c_obj.target_z) > 0.2:
            return True
    return False

def broadcast_message(msg_dict: dict):
    if not connected_clients or not ws_loop:
        return
    msg_str = json.dumps(msg_dict, ensure_ascii=False)
    for ws in list(connected_clients):
        asyncio.run_coroutine_threadsafe(ws.send(msg_str), ws_loop)

latest_frame = None
frame_lock = threading.Lock()

annotated_frame_to_display = None
annotated_lock = threading.Lock()

def ai_worker_thread():
    init_vision_modules()
    
    frame_count = 0
    sam_frame_count = 0
    last_sent_inputs = None
    last_api_call_time = 0.0
    # SAM과 Depth 캐시는 AI 작업 스레드 내부에서 한 세트로 관리합니다.
    sam_cache = {
        "last_labels": [],
        "last_bboxes": [],
        "last_masks_list": [],
        "last_sam_data": [],
        "cached_depth_map": None,
    }

    print("[서버] 백그라운드 AI 비전 파이프라인 가동...")
    while True:
        time.sleep(0.03) # CPU 과점유 방지
        
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_count += 1

        # 5프레임마다 1번씩 AI 처리
        if frame_count % 5 != 0:
            continue

        # 원본 카메라 루프가 아니라 실제 Vision 처리 횟수를 기준으로 SAM 갱신 주기를 계산합니다.
        sam_frame_count += 1
        scene_data = build_scene_graph_for_frame(
            frame,
            frame_count,
            sam_frame_count,
            sam_cache,
        )
        inputs = build_inputs_from_scene(scene_data)

        # 박스 그리기 (obj)
        annotated_frame = frame.copy()
        if inputs:
            for inp in inputs:
                if inp.bbox_2d:
                    x1, y1, x2, y2 = map(int, inp.bbox_2d)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Obj {inp.object_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 메인 쓰레드가 보여줄 화면 업데이트
        with annotated_lock:
            global annotated_frame_to_display
            annotated_frame_to_display = annotated_frame    

        if not inputs:
            continue
            
        # [유니티용] 실시간 좌표 고속 스트림 (6FPS) 전송
        # vlm 판단을 제외한 내용은 반복적으로 전송
        fast_stream_data = [
            {
                "object_id": inp.object_id,
                "target_z": inp.target_z,
                "centroid_y": inp.centroid_y,
                "bbox_2d": inp.bbox_2d
            } for inp in inputs
        ]
        broadcast_message({
            "status": "FAST_STREAM",
            "data": fast_stream_data
        })

        # vlm 호출 판단
        if not is_significant_change(last_sent_inputs, inputs):
            continue
        
        # vlm 호출 쿨타임 : 트리거가 발동해도 10초가 지나지 않았다면 큐에 작업 생성 안함.
        current_time = time.time()
        if current_time - last_api_call_time < 10.0:
            continue

        # Vision worker는 VLM을 직접 호출하지 않고 최신 장면을 작업 큐에 넣은 뒤
        # 즉시 다음 Vision 프레임 처리로 돌아갑니다.
        pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        job = {
            "inputs": inputs,
            "image": pil_image,
            "created_at": current_time,
        }

        # 큐가 차 있다면 아직 처리되지 않은 과거 장면을 버리고 최신 장면으로 교체합니다.
        if vlm_queue.full():
            try:
                vlm_queue.get_nowait()     # 큐에있던 작업을 바로 꺼내버림
            except queue.Empty:         # vlm 스레드에서 작업을 바로 가져갔을 때    
                pass
        vlm_queue.put_nowait(job)       # 작업 바로 push

        # delta trigger와 10초 쿨다운의 기준은 VLM 작업을 큐에 넣은 시점으로 유지합니다.
        last_sent_inputs = inputs
        last_api_call_time = current_time


def vlm_worker_thread():
    """Vision worker와 독립적으로 큐의 최신 장면을 VLM으로 해석합니다."""
    while True:
        # 작업이 없을 때는 blocking 상태로 기다리므로 불필요하게 CPU를 사용하지 않습니다.
        job = vlm_queue.get()
        inputs = job["inputs"]
        pil_image = job["image"]
        batch_input = SemanticInterpretationBatchInput(
            context=DEFAULT_CONTEXT,
            objects=inputs,
        )

        print(f"--- [서버] VLM worker 호출 (객체 {len(inputs)}개) ---")
        vlm_started_at = time.perf_counter()
        try:
            batch_output = interpret_batch(batch_input, image=pil_image)

            broadcast_message({
                "status": "SUCCESS",
                "data": batch_output.model_dump()
            })
            print("-> 유니티로 실시간 좌표+GPT 데이터 전송 완료!")
        except Exception as e:
            print(f"[VLM worker OpenAI GPT 호출 에러] {e}")
            broadcast_message({
                "status": "API_LIMIT_EXCEEDED",
                "data": None
            })
        finally:
            vlm_elapsed = time.perf_counter() - vlm_started_at
            print(f"[서버] VLM worker 판단 소요 시간: {vlm_elapsed:.2f}초")


def main_vision_loop():
    global latest_frame, annotated_frame_to_display, spatial_converter
    stream = WebcamStream()

    
    print("[서버] 실시간 카메라 UI 렌더링 시작...")
    
    try:
        while True:
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            with frame_lock:
                latest_frame = frame
                
            with annotated_lock:
                disp = annotated_frame_to_display if annotated_frame_to_display is not None else frame
                
            # OpenCV 윈도우 렌더링 (블로킹 없이 즉각 실행됨)
            cv2.imshow("Layer 6 - WebSocket Server", disp)
            
            # 여기서 waitKey가 프레임마다 끊임없이 호출되므로 창이 멈추지 않음!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[서버] 'q' 입력 감지. 서버 종료를 예약합니다.")
                break
                
    finally:
        stream.release()
        cv2.destroyAllWindows()

async def ws_handler(websocket):
    connected_clients.add(websocket)
    remote_ip = websocket.remote_address
    print(f"[WebSocket] 유니티 클라이언트 접속됨: {remote_ip}")
    try:
        async for message in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print(f"[WebSocket] 유니티 클라이언트 접속 종료: {remote_ip}")

ws_loop = None

async def run_ws_server():
    print("[서버] WebSocket 서버 시작 대기: ws://127.0.0.1:8765")
    async with websockets.serve(ws_handler, "127.0.0.1", 8765):
        print("[서버] 클라이언트 접속 대기 중...")
        await asyncio.Future()  # 무한 대기

def start_websocket_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(run_ws_server())

if __name__ == "__main__":
    # 1. 백그라운드 쓰레드에서 WebSocket 서버 실행 (비동기)
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()

    # 2. 백그라운드 쓰레드에서 Vision 파이프라인과 FAST_STREAM 생성 실행
    ai_thread = threading.Thread(target=ai_worker_thread, daemon=True)
    ai_thread.start()

    # 3. 별도 백그라운드 쓰레드에서 느린 VLM 추론 실행
    vlm_thread = threading.Thread(target=vlm_worker_thread, daemon=True)
    vlm_thread.start()

    # 4. 메인 쓰레드에서 카메라 화면 띄우기 실행
    # (macOS에서는 cv2.imshow를 반드시 메인 쓰레드에서 호출해야 함)
    main_vision_loop()
