import os
import keras
from utils import load_ubyte_tensors, separate_training_data, get_this_file_dir

X_tensor, y_tensor = load_ubyte_tensors(
    'train/train-images-idx3-ubyte', 
    'train/train-labels-idx1-ubyte'
)

X_train, y_train, X_val, y_val = separate_training_data(X_tensor, y_tensor, 50000)

model: keras.Sequential = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
              
print("\nÎncepe procesul de antrenare...")
model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_val, y_val)
)
model.save(os.path.join(get_this_file_dir(), 'model', 'model.keras'))
print(f"\nModelul a fost salvat cu succes")