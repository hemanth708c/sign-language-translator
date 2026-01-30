import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ===== Load Models =====
letter_model = load_model("models/sign_model.keras")
letter_encoder = joblib.load("models/label_encoder.pkl")

word_model = load_model("models/word_model.h5")
word_labels = ["HELLO", "NAMASTE", "THANKYOU"]  # order from training

# ===== MediaPipe Tasks API =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# ===== Camera =====
cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = landmarker.detect_for_video(
            mp_image,
            int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        if results.hand_landmarks:

            h, w, _ = frame.shape
            hand_count = len(results.hand_landmarks)

            # ===== Draw dots on hands =====
            for hand in results.hand_landmarks:
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0,255,0), -1)

            # ===== CASE 1: ONE HAND → LETTER MODEL =====
            if hand_count == 1:
                hand = results.hand_landmarks[0]
                row = []
                for lm in hand:
                    row += [lm.x, lm.y, lm.z]

                X = np.array(row).reshape(1,63)
                prediction = letter_model.predict(X, verbose=0)
                predicted_class = np.argmax(prediction)
                letter = letter_encoder.inverse_transform([predicted_class])[0]

                cv2.putText(frame, letter, (30,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

            # ===== CASE 2: TWO HANDS → WORD MODEL =====
            elif hand_count == 2:
                row = []
                for hand in results.hand_landmarks:
                    for lm in hand:
                        row += [lm.x, lm.y, lm.z]

                X = np.array(row).reshape(1,126)
                prediction = word_model.predict(X, verbose=0)
                word_index = np.argmax(prediction)
                word = word_labels[word_index]

                cv2.putText(frame, word, (30,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
