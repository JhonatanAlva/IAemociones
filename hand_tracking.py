import cv2
import mediapipe as mp
import pyautogui
import time

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.screen_w, self.screen_h = pyautogui.size()
        self.last_click_time = time.time()
        self.last_mouse_x = None
        self.last_mouse_y = None

    def smooth_move(self, target_x, target_y, factor=0.3):
        if self.last_mouse_x is None:
            self.last_mouse_x = target_x
            self.last_mouse_y = target_y
        smooth_x = int(self.last_mouse_x + factor * (target_x - self.last_mouse_x))
        smooth_y = int(self.last_mouse_y + factor * (target_y - self.last_mouse_y))
        self.last_mouse_x, self.last_mouse_y = smooth_x, smooth_y
        return smooth_x, smooth_y

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return

        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            index = landmarks[8]
            middle = landmarks[12]
            thumb = landmarks[4]

            screen_x = int(index.x * self.screen_w)
            screen_y = int(index.y * self.screen_h)

            # Movimiento solo si cambia lo suficiente
            if self.last_mouse_x is None or abs(screen_x - self.last_mouse_x) > 10 or abs(screen_y - self.last_mouse_y) > 10:
                smooth_x, smooth_y = self.smooth_move(screen_x, screen_y)
                pyautogui.moveTo(smooth_x, smooth_y)

            current_time = time.time()

            # Click
            if index.y > middle.y and current_time - self.last_click_time > 0.5:
                pyautogui.click()
                self.last_click_time = current_time

            # Scroll
            elif index.y < thumb.y and middle.y < thumb.y:
                pyautogui.scroll(10 if index.y < middle.y else -10)

            # Zoom in
            elif abs(index.x - thumb.x) < 0.03:
                pyautogui.hotkey('ctrl', '+')
                time.sleep(0.2)

            # Zoom out
            elif abs(index.x - thumb.x) > 0.1:
                pyautogui.hotkey('ctrl', '-')
                time.sleep(0.2)
