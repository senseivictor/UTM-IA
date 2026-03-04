import os
from tensorflow import config
from utils import (
    load_images_and_masks_tensors, 
    get_this_file_dir, 
    separate_training_data,
    simple_unet
)

print("Numar GPU-uri disponibile: ", len(config.list_physical_devices('GPU')))

X_tensor, y_tensor = load_images_and_masks_tensors(
    "./data/CameraRGB/", 
    "./data/CameraMask/"
)

X_train, y_train, X_val, y_val = separate_training_data(X_tensor, y_tensor, 680)

model = simple_unet()

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

model.save(os.path.join(get_this_file_dir(), 'model', 'unet_model.h5'))