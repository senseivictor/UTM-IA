import os
import pandas as pd
import numpy as np
import keras
from utils import load_ubyte

X_train_full_tensor, y_train_full_tensor = load_ubyte(
    'train-images-idx3-ubyte', 
    'train-labels-idx1-ubyte'
)

# Datele pentru antrenare
X_train: np.ndarray = X_train_full_tensor[:50000].astype("float32") / 255.0
y_train: np.ndarray = y_train_full_tensor[:50000]

# Datele pentru validare
X_valid: np.ndarray = X_train_full_tensor[50000:].astype("float32") / 255.0
y_valid: np.ndarray = y_train_full_tensor[50000:]

# Definirea Arhitecturii
model: keras.Sequential = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 4. Compilarea
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Antrenarea
print("\nÎncepe procesul de antrenare...")
model.fit(X_train, y_train, 
          epochs=10, 
          batch_size=32, 
          validation_data=(X_valid, y_valid))

# 6. Salvarea modelului (Exportăm "cunoștințele")
model.save(model_save_path)
print(f"\nModelul a fost salvat cu succes în '{model_save_path}'")