import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller
import time

# 初始化鍵盤控制器
keyboard = Controller()

# 初始化 MediaPipe 手部模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 用於儲存前一幀的食指位置
prev_y = None
# 設置移動閾值（像素）
threshold = 30
# 冷卻時間（秒）
cooldown = 0.5
last_action_time = 0

# 啟動攝影機
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # 翻轉影像（左右鏡像）
    frame = cv2.flip(frame, 1)
    # 轉換色彩空間 BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 處理影像並偵測手部
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 繪製手部關鍵點
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 獲取食指指尖座標 (INDEX_FINGER_TIP = 8)
            index_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            curr_x, curr_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # 如果有前一幀位置，進行比較
            if prev_y is not None:
                current_time = time.time(0.1)
                # 檢查冷卻時間
                if current_time - last_action_time >= cooldown:
                    diff_y = prev_y - curr_y
                    
                    # 向上移動（y座標變小）
                    if diff_y > threshold:
                        keyboard.press(Key.up)
                        
                        keyboard.release(Key.up)
                        print("按下 '上' 鍵")
                        last_action_time = current_time
                        
                    # 向下移動（y座標變大）
                    elif diff_y < -threshold:
                        keyboard.press(Key.down)
                        
                        keyboard.release(Key.down)
                        print("按下 '下' 鍵")
                        last_action_time = current_time
                    
            
            # 更新前一幀位置
            prev_y = curr_y
            
            # 在畫面上顯示食指位置
            cv2.circle(frame, (curr_x, curr_y), 10, (0, 255, 0), -1)
    
    # 顯示影像
    cv2.imshow('Hand Tracking', frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
hands.close()