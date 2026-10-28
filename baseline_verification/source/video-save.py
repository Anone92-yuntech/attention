import cv2
import numpy as np
import mediapipe as mp
import collections
from ultralytics import YOLO

# ==========================================
# 1. 參數設定與路徑
# ==========================================
VIDEO_PATH = "DD-04-PZ-ST0053-20201204.mov"                                      # 輸入影片路徑
MODEL_PATH = r"D:\johnson\J.H\runs\detect\train-7\weights\best.pt"  # 訓練好的 YOLO11 模型
OUTPUT_VIDEO_PATH = "output_demo.mp4"                              # 輸出展示影片名稱

def calculate_iou(boxA, boxB):
    """計算兩個 Bounding Box 的交集率 (IoU)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def get_hand_bbox(landmarks, img_w, img_h, padding=0.05):
    """將 MediaPipe 手部關鍵點轉換為向外擴展 5% 的 Bounding Box"""
    x_coords = [lm.x * img_w for lm in landmarks.landmark]
    y_coords = [lm.y * img_h for lm in landmarks.landmark]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    w = x_max - x_min
    h = y_max - y_min
    
    x_min -= w * padding
    x_max += w * padding
    y_min -= h * padding
    y_max += h * padding
    
    x_min, x_max = max(0, x_min), min(img_w - 1, x_max)
    y_min, y_max = max(0, y_min), min(img_h - 1, y_max)
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def is_inside_roi(bbox, roi_rect):
    """判斷物件中心點是否在 ROI 內"""
    if roi_rect is None or roi_rect == (0, 0, 0, 0):
        return True
        
    rx, ry, rw, rh = roi_rect
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    
    return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)

def main():
    # 初始化 MediaPipe Holistic (全骨架 + 臉部網格 + 雙手)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("載入 YOLO11 模型中...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"錯誤: 無法開啟影片 {VIDEO_PATH}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 建立 VideoWriter 輸出影片物件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (img_w, img_h))
    
    BUFFER_FRAMES = int(fps * 0.5)  # 0.5 秒歷史緩衝區
    yolo_history = collections.deque(maxlen=BUFFER_FRAMES)
    
    roi_defined = False
    roi_rect = None
    
    cv2.namedWindow("Toy Tracking & Analysis - Demo Export", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Toy Tracking & Analysis - Demo Export", 1280, 720)
    
    print("開始處理並錄製影片...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_results = holistic.process(rgb_frame)
        
        # 第一次偵測到人物時暫停供框選受測者 ROI
        if not roi_defined and (holistic_results.pose_landmarks or holistic_results.left_hand_landmarks or holistic_results.right_hand_landmarks):
            print("偵測到受測者！請於視窗中框選範圍後按 Space 或 Enter 鍵。")
            roi_rect = cv2.selectROI("Toy Tracking & Analysis - Demo Export", frame, showCrosshair=True, fromCenter=False)
            roi_defined = True
            print(f"追蹤範圍已鎖定: {roi_rect}")
            
        current_toys_bboxes = []
        is_playing_current_frame = False
        
        # 繪製 ROI 邊界 (黃色線，加粗至 4)
        if roi_defined and roi_rect != (0, 0, 0, 0):
            rx, ry, rw, rh = roi_rect
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 4, cv2.LINE_AA)
            
        # ------------------------------------------
        # A. YOLO 推論與過濾
        # ------------------------------------------
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                t_bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                if is_inside_roi(t_bbox, roi_rect):
                    current_toys_bboxes.append(t_bbox)
                    # 繪製玩具框 (綠色，粗細 4)
                    cv2.rectangle(frame, (t_bbox[0], t_bbox[1]), (t_bbox[2], t_bbox[3]), (0, 255, 0), 4)
                    
        yolo_history.append(current_toys_bboxes)
        
        # ------------------------------------------
        # B. 繪製骨架 (Face Mesh + Pose + Hands)
        # ------------------------------------------
        # 1. 臉部網格
        if holistic_results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                holistic_results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        # 2. 身體骨架
        if holistic_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                holistic_results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # 3. 手部關鍵點與 BBox
        hand_bboxes = []
        detected_hands = []
        if holistic_results.left_hand_landmarks:
            detected_hands.append(holistic_results.left_hand_landmarks)
        if holistic_results.right_hand_landmarks:
            detected_hands.append(holistic_results.right_hand_landmarks)

        for hand_lms in detected_hands:
            h_bbox = get_hand_bbox(hand_lms, img_w, img_h, padding=0.05)
            if is_inside_roi(h_bbox, roi_rect):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                )
                hand_bboxes.append(h_bbox)
                # 繪製手部擴展框 (藍色，粗細 4)
                cv2.rectangle(frame, (h_bbox[0], h_bbox[1]), (h_bbox[2], h_bbox[3]), (255, 0, 0), 4)
                
        # ------------------------------------------
        # C. 觸發條件判定 (有任何交集 iou > 0 且包含 0.5s 殘影)
        # ------------------------------------------
        for h_bbox in hand_bboxes:
            for history_frame_boxes in yolo_history:
                for t_bbox in history_frame_boxes:
                    iou = calculate_iou(h_bbox, t_bbox)
                    if iou > 0:
                        is_playing_current_frame = True
                        break
                if is_playing_current_frame:
                    break
            if is_playing_current_frame:
                break
                
        # ------------------------------------------
        # D. 右上角即時狀態標示 (超大字體 10.0 自動靠右上對齊)
        # ------------------------------------------
        status_text = "playing" if is_playing_current_frame else "none"
        text_color = (0, 255, 255) if is_playing_current_frame else (0, 0, 255)
        font_scale = 10.0
        thickness = 20

        # 動態量測文字長寬
        (text_width, text_height), baseline = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        margin = 30
        text_x = img_w - text_width - margin
        text_y = text_height + margin

        cv2.putText(
            frame, 
            status_text, 
            (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            text_color, 
            thickness, 
            cv2.LINE_AA
        )
            
        # 寫入影格至輸出影片檔
        out.write(frame)
        
        cv2.imshow("Toy Tracking & Analysis - Demo Export", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"展示影片匯出完成！檔案已儲存為: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()