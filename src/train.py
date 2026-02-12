import os
import pandas as pd
import numpy as np
import keras

# Obținem calea absolută către folderul unde se află scriptul curent
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
csv_train_path: str = os.path.join(BASE_DIR, 'data', 'fashion-mnist_train.csv')
model_save_path = os.path.join(BASE_DIR, 'model', 'model_fashion.h5')

print("Se încarcă datele pentru antrenare...")
train_df: pd.DataFrame = pd.read_csv(csv_train_path)

# Extragem etichetele și pixelii
y_train_full: np.ndarray = train_df['label'].values
X_train_full: np.ndarray = train_df.drop('label', axis=1).values.reshape(-1, 28, 28)

# 2. Pregătirea datelor (Slicing & Normalizare)
X_train: np.ndarray = X_train_full[:50000].astype("float32") / 255.0
X_valid: np.ndarray = X_train_full[50000:].astype("float32") / 255.0
y_train: np.ndarray = y_train_full[:50000]
y_valid: np.ndarray = y_train_full[50000:]

# 3. Definirea Arhitecturii
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