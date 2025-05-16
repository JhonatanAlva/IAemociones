import cv2
import time
from hand_tracking import HandTracker
from emotion_detection import detect_emotion, last_region, apply_filter

def get_emotion_color(emotion):
    colors = {
        "happy": (0, 255, 0),
        "sad": (255, 0, 0),
        "angry": (0, 0, 255),
        "surprise": (255, 255, 0),
        "fear": (255, 0, 255),
        "neutral": (128, 128, 128),
        "disgust": (0, 255, 255)
    }
    return colors.get(emotion, (255, 255, 255))

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker()

    if not cap.isOpened():
        print("Error: no se pudo abrir la camara.")
        return

    frame_count = 0
    FRAME_SKIP = 10  # analiza cada 10 frames
    emotion = None
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        tracker.process(frame)

        # Solo analiza emociones cada FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            emotion = detect_emotion(frame)
        frame_count += 1

        # Dibuja región y aplica filtro visual en el hilo principal
        if last_region and emotion:
            frame[:] = apply_filter(frame, emotion)
            x, y, w, h = last_region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Dibujar borde del marco según emoción
        color = get_emotion_color(emotion) if emotion else (255, 255, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)

        # Mensaje motivacional
        if emotion in ["angry", "sad"]:
            cv2.putText(frame, "Animo Todo estara bien :)",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Rastreo de Manos + Deteccion de Emocion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
