import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load WITHOUT header
data = pd.read_csv("dataset/gesture_words.csv", header=None)

# Last column is label
X = data.iloc[:, :-1].values   # all except last
y = data.iloc[:, -1].values    # last column

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

model = Sequential([
    Dense(128, activation='relu', input_shape=(126,)),
    Dense(64, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=30)

model.save("models/word_model.h5")
print("Word model saved!")
print("Classes:", encoder.classes_)
