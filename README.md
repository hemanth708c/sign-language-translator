
---

# Sign Language Translator

A real-time **Sign Language Translator** built using **MediaPipe hand landmarks** and **deep learning**.
The system captures hand gestures from a webcam, extracts landmark features, and predicts corresponding **alphabets and words**.

---

## Overview

This project focuses on **landmark-based gesture recognition**, making it more robust to background noise, lighting variations, and camera quality compared to raw image-based models.

The model is trained on custom-collected hand landmark data and supports both **single-hand** and **multi-hand** gestures.

---

## Key Features

* Real-time gesture recognition using webcam
* MediaPipe Hand Landmarks for precise hand tracking
* Deep learning models trained on custom datasets
* Alphabet (A–Z) recognition
* Word-level gesture support (e.g., Hello, Thank You, Namaste)
* Modular pipeline for data collection, training, and inference

---

## Technologies Used

* Python
* OpenCV
* MediaPipe (Tasks API)
* TensorFlow / Keras
* NumPy
* Pandas

---

## Project Structure

```
sign_language_translator/
│
├── app/
│   └── predict.py                  # Real-time gesture prediction
│
├── dataset/
│   ├── images/
│   ├── gesture_data.csv            # Alphabet landmarks
│   └── gesture_words.csv           # Word landmarks
│
├── models/
│   ├── sign_model.h5
│   ├── sign_model_landmarks.keras
│   └── word_model.h5
│
├── capture_data.py                 # Capture alphabet gestures
├── capture_words.py                # Capture word gestures
├── train_model.py                  # Train alphabet model
├── train_words.py                  # Train word model
├── hand_landmarker.task
└── README.md
```

---

## How It Works

1. Webcam captures live video frames.
2. MediaPipe detects hand landmarks (x, y, z coordinates).
3. Landmark vectors are passed to a trained neural network.
4. The model predicts the corresponding alphabet or word.
5. The prediction is displayed in real time.

---

## Installation

```bash
pip install -r requirements.txt
```

Ensure your system camera is enabled and accessible.

---

## Running the Project

```bash
python app/predict.py
```

Press **Q** to exit the application.

---

## Training Custom Gestures

* Run `capture_data.py` to collect alphabet gestures.
* Run `train_model.py` to train the alphabet classifier.
* Use `capture_words.py` and `train_words.py` for word-level gestures.

All data is stored as landmark coordinates for consistency and accuracy.

---

## Limitations

* Dynamic gestures require multiple frames for stable prediction.
* Two-hand gestures require additional temporal modeling for higher accuracy.
* Currently optimized for controlled environments.

---

## Future Enhancements

* Sentence-level gesture recognition
* Temporal models (LSTM / Transformer) for dynamic signs
* Text-to-speech output
* Mobile and web deployment
* Expanded vocabulary

---

## Author

**Hemanth J**
Computer Science Engineering
Focus areas: Artificial Intelligence, Machine Learning, Computer Vision

---

