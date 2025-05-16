import cv2
from deepface import DeepFace
from datetime import datetime
import csv
import os
import pyttsx3
import threading
import time

# Variables globales
LOG_FILE = "emotions_log.csv"
DETECTED = {}
LAST_CAPTURED_EMOTION = None
last_emotion = None
last_region = None
analyzing = False

# Cola manual de mensajes de voz
voice_messages = []
voice_lock = threading.Lock()

# Hilo de voz único
def voice_loop():
    while True:
        if voice_messages:
            with voice_lock:
                emotion, mensaje = voice_messages.pop(0)
            try:
                engine = pyttsx3.init()
                engine.say(mensaje)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[Error] Voz emoción '{emotion}':", e)
        else:
            time.sleep(0.1)

# Iniciar el hilo de voz al arrancar
threading.Thread(target=voice_loop, daemon=True).start()

def speak_emotion(emotion):
    mensajes = {
        "happy": "¡Veo que estás feliz! Sigue así.",
        "sad": "Parece que estás triste. ¡Ánimo!",
        "angry": "Tranquilo, respira profundo.",
        "surprise": "¡Sorpresa detectada!",
        "fear": "Todo va a estar bien.",
        "neutral": "Todo está tranquilo por ahora."
    }

    if emotion not in DETECTED or (datetime.now() - DETECTED[emotion]).seconds > 10:
        mensaje = mensajes.get(emotion, "")
        if mensaje:
            with voice_lock:
                voice_messages.append((emotion, mensaje))
        DETECTED[emotion] = datetime.now()
        print(f"[Voz] Emoción detectada: {emotion}")

def log_emotion(emotion):
    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "emotion"])
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion])

def apply_filter(frame, emotion):
    filters = {
        "happy": cv2.COLORMAP_SUMMER,
        "sad": cv2.COLORMAP_BONE,
        "angry": cv2.COLORMAP_HOT,
        "surprise": cv2.COLORMAP_OCEAN,
        "fear": cv2.COLORMAP_PINK
    }
    if emotion in filters:
        return cv2.applyColorMap(frame, filters[emotion])
    return frame

def save_emotion_capture(frame, emotion):
    global LAST_CAPTURED_EMOTION
    if emotion in ["happy", "sad", "angry"] and emotion != LAST_CAPTURED_EMOTION and last_region:
        x, y, w, h = last_region
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        folder = "capturas_emociones"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/capture_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{emotion}.jpg"
        cv2.imwrite(filename, frame)
        LAST_CAPTURED_EMOTION = emotion

def analyze_emotion_in_background(original_frame):
    global last_emotion, last_region, analyzing

    if analyzing:
        return

    analyzing = True
    try:
        resized = cv2.resize(original_frame, (0, 0), fx=0.3, fy=0.3)
        results = DeepFace.analyze(resized, actions=['emotion'], enforce_detection=False)

        for face in results:
            emotion = face['dominant_emotion']
            region = face['region']
            x, y, w, h = [int(v * (1 / 0.3)) for v in (region['x'], region['y'], region['w'], region['h'])]

            if x > 0 and y > 0:
                last_emotion = emotion
                last_region = (x, y, w, h)

                speak_emotion(emotion)
                log_emotion(emotion)
                apply_filter(original_frame.copy(), emotion)
                save_emotion_capture(original_frame.copy(), emotion)

    except Exception as e:
        print("Error detecting emotion:", e)
    finally:
        analyzing = False

def detect_emotion(frame):
    threading.Thread(target=analyze_emotion_in_background, args=(frame.copy(),), daemon=True).start()
    return last_emotion

# Función reutilizable para entorno web
def detect_emotion_from_frame(frame):
    global last_emotion, last_region, analyzing

    if analyzing:
        return last_emotion

    analyzing = True
    try:
        resized = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
        results = DeepFace.analyze(resized, actions=['emotion'], enforce_detection=False)

        for face in results:
            emotion = face['dominant_emotion']
            region = face['region']
            x, y, w, h = [int(v * (1 / 0.3)) for v in (region['x'], region['y'], region['w'], region['h'])]

            if x > 0 and y > 0:
                last_emotion = emotion
                last_region = (x, y, w, h)

                speak_emotion(emotion)
                log_emotion(emotion)
                apply_filter(frame.copy(), emotion)
                save_emotion_capture(frame.copy(), emotion)

        return last_emotion
    except Exception as e:
        print("Error en detección web:", e)
        return None
    finally:
        analyzing = False
