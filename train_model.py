import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os

data = pd.read_csv("dataset/gesture_data.csv")

X = data.drop("label", axis=1).values
y = data["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

os.makedirs("models", exist_ok=True)
model.save("models/sign_model.keras")
joblib.dump(encoder, "models/label_encoder.pkl")

print("Model trained and saved!")
