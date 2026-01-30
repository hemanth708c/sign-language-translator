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

dataset_path = "dataset/gesture_words.csv"
os.makedirs("dataset", exist_ok=True)

# Write header if file not exists
if not os.path.exists(dataset_path):
    with open(dataset_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(126):  # 42 landmarks × 3
            header.append(f"f{i}")
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

        if results.hand_landmarks and len(results.hand_landmarks)==2:
            # Draw dots
            h, w, _ = frame.shape
            for hand in results.hand_landmarks:
                for lm in hand:
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    cv2.circle(frame,(cx,cy),3,(0,255,0),-1)

        cv2.imshow("Collect Words Dataset", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press keys to save samples
        if key in [ord('h'), ord('t'), ord('n')]:
            if results.hand_landmarks and len(results.hand_landmarks)==2:
                row = []
                for hand in results.hand_landmarks:
                    for lm in hand:
                        row += [lm.x, lm.y, lm.z]

                label = {ord('h'):"HELLO", ord('t'):"THANKYOU", ord('n'):"NAMASTE"}[key]
                row.append(label)

                with open(dataset_path,'a',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print("Saved:", label)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
