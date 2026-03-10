import os
from scripts.utils import (
    load_images_and_masks_tensors, 
    separate_training_data,
    simple_unet,
    models_path
)

X_tensor, y_tensor = load_images_and_masks_tensors(
    __file__,
    'resized/CameraRGB', 
    'resized/CameraMask'
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
    batch_size=16,
    validation_data=(X_val, y_val)
)

model.save(models_path(__file__) / 'unet_model.h5')