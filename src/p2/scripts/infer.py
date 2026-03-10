import tensorflow as tf
import matplotlib.pyplot as plt
import os
import keras
from scripts.utils import load_images_and_masks_tensors, load_project_model, data_path

unet: keras.Model = load_project_model(__file__, 'unet_model.h5')

X_images, y_masks = load_images_and_masks_tensors(
    __file__, 
    'resized/CameraRGB', 
    'resized/CameraMask', 
    selection=slice(-3, None)
)

pred_masks_raw = unet.predict(X_images)

for idx in range(X_images.shape[0]):
    
    # Procesăm predicția curentă
    pred_mask_processed = tf.argmax(pred_masks_raw[idx], axis=-1)
    
    # Pregătim figura
    plt.figure(figsize=(15, 4))
    
    # Subplot 1: Imaginea Originală
    plt.subplot(1, 3, 1)
    plt.title(f"Imaginea {idx+1} - Original")
    plt.imshow(X_images[idx])
    plt.axis('off')

    # Subplot 2: Masca Reală
    plt.subplot(1, 3, 2)
    plt.title("Masca Reală")
    # .numpy() transformă tensorul în array, squeeze() elimină canalul de 1
    plt.imshow(y_masks[idx].numpy().squeeze())
    plt.axis('off')

    # Subplot 3: Predicția
    plt.subplot(1, 3, 3)
    plt.title("Predicția Modelului")
    plt.imshow(pred_mask_processed)
    plt.axis('off')

    plt.tight_layout()
    plt.show() # Afișează figura curentă înainte de a trece la următoarea