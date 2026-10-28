import cv2
import numpy as np
import mediapipe as mp
import collections
import csv
from ultralytics import YOLO

# ==========================================
# 1. 參數設定與初始化
# ==========================================
VIDEO_PATH = r"DD-22-PZ-ST0022-20210325.mp4"    # 請替換成你的影片檔案路徑
MODEL_PATH = r"C:\Users\User\Desktop\johnson\attention\best.pt"  # 請替換成你訓練好的 YOLO11 模型路徑
OUTPUT_CSV = "action_log.csv"    # 輸出的 CSV 檔案名稱

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
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_hand_bbox(landmarks, img_w, img_h, padding=0.05):
    """將 MediaPipe 手部關鍵點轉換為擴展後的 Bounding Box"""
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
    """判斷物件中心點是否在使用者框選的 ROI 範圍內"""
    if roi_rect is None or roi_rect == (0, 0, 0, 0):
        return True
        
    rx, ry, rw, rh = roi_rect
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    
    return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)

def main():
    # 改用 MediaPipe Holistic (同時偵測臉部、身體與手部)
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
        print(f"錯誤: 無法讀取影片 {VIDEO_PATH}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    BUFFER_FRAMES = int(fps * 0.5)
    WINDOW_FRAMES = int(fps * 5)
    
    yolo_history = collections.deque(maxlen=BUFFER_FRAMES)
    
    frame_count = 0
    trigger_count = 0
    window_start_sec = 0
    
    roi_defined = False
    roi_rect = None
    
    cv2.namedWindow("Toy Tracking & Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Toy Tracking & Analysis", 1280, 720)
    
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_Window", "Status"])
        
        print("開始處理影片...")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            holistic_results = holistic.process(rgb_frame)
            
            # 第一幀偵測到身體/人物姿態時暫停供框選 ROI
            if not roi_defined and (holistic_results.pose_landmarks or holistic_results.left_hand_landmarks or holistic_results.right_hand_landmarks):
                print("偵測到人員！請在視窗中框選受測者範圍後按 Space 或 Enter。")
                roi_rect = cv2.selectROI("Toy Tracking & Analysis", frame, showCrosshair=True, fromCenter=False)
                roi_defined = True
                print(f"已設定追蹤範圍: {roi_rect}")
                
            frame_count += 1
            current_toys_bboxes = []
            is_playing_current_frame = False
            
            if roi_defined and roi_rect != (0, 0, 0, 0):
                rx, ry, rw, rh = roi_rect
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2, cv2.LINE_AA)
            
            # ------------------------------------------
            # A. YOLO 推論與過濾
            # ------------------------------------------
            results = model(frame, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    t_bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    if is_inside_roi(t_bbox, roi_rect):
                        current_toys_bboxes.append(t_bbox)
                        cv2.rectangle(frame, (t_bbox[0], t_bbox[1]), (t_bbox[2], t_bbox[3]), (0, 255, 0), 2)
                    
            yolo_history.append(current_toys_bboxes)
            
            # ------------------------------------------
            # B. 繪製全身骨架 (Face + Pose + Hands)
            # ------------------------------------------
            # 繪製臉部網格
            if holistic_results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    holistic_results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # 繪製身體骨架
            if holistic_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    holistic_results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # 提取並繪製手部
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
                    cv2.rectangle(frame, (h_bbox[0], h_bbox[1]), (h_bbox[2], h_bbox[3]), (255, 0, 0), 2)
                    
            # ------------------------------------------
            # C. 觸發條件判定 (只要有交集就觸發)
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
            # D. 右上角即時狀態顯示
            # ------------------------------------------
            if is_playing_current_frame:
                trigger_count += 1
                cv2.putText(frame, "playing", (img_w - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
            else:
                cv2.putText(frame, "none", (img_w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
            # ------------------------------------------
            # E. 5 秒時間窗口 CSV 結算
            # ------------------------------------------
            if frame_count % WINDOW_FRAMES == 0:
                window_end_sec = window_start_sec + 5
                time_label = f"{window_start_sec:02d}s - {window_end_sec:02d}s"
                
                if trigger_count >= 3:
                    writer.writerow([time_label, "playing"])
                else:
                    writer.writerow([time_label, "none"])
                    
                trigger_count = 0
                window_start_sec += 5
                
            cv2.imshow("Toy Tracking & Analysis", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"處理完成！結果已儲存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()