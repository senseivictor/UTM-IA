import os
import pandas as pd
import numpy as np
import struct
import tensorflow as tf


def get_this_file_dir():
   return os.path.dirname(os.path.abspath(__file__))

def process_path(image_path, mask_path):
    # 1. Citim fișierul de pe disc
    img = tf.io.read_file(image_path)
    # 2. Îl transformăm în pixeli (decodare .png)
    img = tf.image.decode_png(img, channels=3)
    # 3. Normalizăm (facem pixelii mici, între 0 și 1) - ajută NIU (n) să lucreze mai bine
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Facem același lucru pentru mască
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1) # Măștile sunt adesea grayscale (1 canal)
    
    # Redimensionăm ambele la aceeași mărime (ex: 128x128)
    img = tf.image.resize(img, (128, 128))
    mask = tf.image.resize(mask, (128, 128))

    return img, mask

def simple_unet(input_shape=(128, 128, 3), num_classes=23):
    inputs = tf.keras.layers.Input(input_shape)

    # --- ENCODER (Contracting Path) ---
    # Strat 1: Imaginea e mare, modelul caută linii/margini
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) # Imaginea devine 64x64

    # Strat 2: Imaginea e mai mică, modelul caută forme (roți, ferestre)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2) # Imaginea devine 32x32

    # --- BOTTOM (The Bridge) ---
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)

    # --- DECODER (Expanding Path) ---
    # Strat 3: Mărim imaginea înapoi
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3) # Revine la 64x64
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up1)

    # Strat 4: Revenim la dimensiunea originală
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4) # Revine la 128x128
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up2)

    # --- OUTPUT ---
    # Ultimul strat: Softmax ne dă probabilitatea pentru fiecare clasă per pixel
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(conv5)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = simple_unet()