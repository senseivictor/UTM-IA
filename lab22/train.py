import tensorflow as tf
from tensorflow import config
import numpy as np
import os
from utils import process_path, model, get_this_file_dir

print("Numar GPU-uri disponibile: ", len(config.list_physical_devices('GPU')))

path = get_this_file_dir()
image_path = os.path.join(path, './data/CameraRGB/')
mask_path = os.path.join(path, './data/CameraMask/')

image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path)])
mask_files = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path)])

# 2. Slicing pentru Training (80% din ~850 de imagini = 680)
train_images = image_files[:680]
train_masks = mask_files[:680]

# 3. CREAREA OBIECTULUI DATASET (Aici era eroarea!)
# Transformăm listele de string-uri într-un obiect tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))

model.compile(
    optimizer='adam', # Acesta folosește NIU (learning rate) pentru a ajusta greutățile
    loss='sparse_categorical_crossentropy', # Calculează eroarea pentru fiecare pixel
    metrics=['accuracy']
)

# Împărțim datele în grupuri (batch-uri) de 32
dataset = dataset.map(process_path).batch(32)

# Pornim antrenarea
model.fit(
    dataset, 
    epochs=10,
)

model.save(os.path.join(get_this_file_dir(), 'model', 'unet_model.h5'))