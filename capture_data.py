import cv2
import mediapipe as mp
import csv
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

dataset_path = "dataset/gesture_data.csv"
os.makedirs("dataset", exist_ok=True)

# Create CSV header if not exists
if not os.path.exists(dataset_path):
    with open(dataset_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f'x{i}', f'y{i}', f'z{i}']
        header.append("label")
        writer.writerow(header)

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
            for lm in results.hand_landmarks[0]:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx,cy), 3, (0,255,0), -1)

        cv2.putText(frame,
                    "Press letter key to save sample",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

        cv2.imshow("Collect Data", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press letter key to save
        if key >= ord('a') and key <= ord('z'):
            if results.hand_landmarks:
                row = []
                for lm in results.hand_landmarks[0]:
                    row += [lm.x, lm.y, lm.z]
                row.append(chr(key).upper())

                with open(dataset_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print("Saved:", chr(key).upper())

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
