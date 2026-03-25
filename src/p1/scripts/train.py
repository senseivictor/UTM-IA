import keras
from keras import layers
from scripts.utils import load_ubyte_tensors, separate_training_data, models_path

X_tensor, y_tensor = load_ubyte_tensors(
    __file__,
    'train/train-images-idx3-ubyte', 
    'train/train-labels-idx1-ubyte'
)

X_train, y_train, X_val, y_val = separate_training_data(X_tensor, y_tensor, 50000)

model: keras.Sequential = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
              
model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_val, y_val)
)
model.save(models_path(__file__) / "model.keras")