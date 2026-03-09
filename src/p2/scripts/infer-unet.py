import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import load_images_and_masks_tensors, get_this_file_dir

# 1. Căi către date și model
path = get_this_file_dir()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'unet_model.h5')
image_dir = os.path.join(path, '..', 'data', 'resized', 'CameraRGB')
mask_dir = os.path.join(path, '..', 'data', 'resized', 'CameraMask')

# 2. Încărcăm modelul
print("Se încarcă modelul...")
unet = tf.keras.models.load_model(model_path)

# 3. Luăm o imagine de test (una de la finalul listei, nefolosită în training)
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

test_img_path = image_files[-1] # Ultima imagine din dataset
test_mask_path = mask_files[-1]

# Procesăm imaginea pentru model
image, mask = load_images_and_masks_tensors(test_img_path, test_mask_path)

# 4. Predicția
# Adăugăm o dimensiune extra (batch size) pentru că modelul așteaptă (1, 128, 128, 3)
pred_mask = unet.predict(image[tf.newaxis, ...])

# Luăm clasa cu probabilitatea maximă pentru fiecare pixel (Argmax)
pred_mask = tf.argmax(pred_mask, axis=-1)
pred_mask = pred_mask[0] # Scoatem dimensiunea de batch

# 5. Vizualizare
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Imagine Originală")
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.title("Masca Reală (Ground Truth)")
plt.imshow(mask)

plt.subplot(1, 3, 3)
plt.title("Predicția Modelului")
plt.imshow(pred_mask)

plt.show()