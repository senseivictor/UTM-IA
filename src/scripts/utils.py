import sys
import os
import pandas as pd
import numpy as np
import struct
import tensorflow as tf
from pathlib import Path
from typing import Union, Any, cast
import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def subproject_root(file: str):
   """
   Urcă în ierarhie până găsește folderul care conține 
   atât subfolderul 'data' cât și 'scripts'.
   """
   current_path = Path(file).resolve()
   
   # Parcurgem toți părinții fișierului curent
   for parent in current_path.parents:
      # Verificăm existența ambelor directoare în locația curentă
      if (parent / "data").is_dir() and (parent / "scripts").is_dir():
         return parent
         
   # Fallback: returnează directorul părinte dacă nu găsește structura
   return current_path.parent

def data_path(file):
   return subproject_root(file) / 'data'

def models_path(file):
   return subproject_root(file) / 'models'

def load_project_model(file, model_name):
   path = models_path(file) / model_name
   loaded_model = keras.models.load_model(path)
   return cast(Any, loaded_model)

def load_ubyte_tensors(file, images_path: Union[str, Path], labels_path: Union[str, Path]):
   full_images_path = data_path(file) / images_path
   full_labels_path = data_path(file) / labels_path
   with open(full_images_path, 'rb') as f:
      # Citim header-ul (primii 16 octeti (bytes)): 
      #  - Magic Number (4 bytes): 
      #      ex. 0x00 00 08 03
      #      * primii doi octeti sunt mereu 00 00
      #      * al treilea octet este 08 (codul pentru unsigned integer)
      #      * al patrulea octet este dimensiunea (de exemplu 03 pentru un tensor de 3 dimensiuni)
      #      *** magic number e doar pentru verificare daca e valid 
      #           (de obicei in zecimal e 2049 sau 2051 pentru fisierele ubyte). 
      #           In codul acesta NU verificam asta
      #  - Număr Imagini (4 bytes)
      #  - Rânduri (4 bytes)
      #  - Coloane (4 bytes)
      # '>IIII' înseamnă: Big-Endian, 4 numere întregi nesemnate (unsigned int)
      # Adica diferenta dintre csv si ubyte este ca ubyte nu stocheaza virgulele, 
      # dar codul binar in sine, iar toate datele de care are nevoie pentru reshaping,
      # se afla in header
      magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
      
      # Citim restul fișierului ca un șir de octeți (uint8)
      raw_images_bytes = np.fromfile(f, dtype=np.uint8)
      
      # Transformăm șirul liniar în Tensor 3D (num, rows, cols)
      X_images_full_tensor = raw_images_bytes.reshape(num, rows, cols)

   with open(full_labels_path, 'rb') as f:
      # Citim header-ul: Magic Number (4 bytes), Număr Itemi (4 bytes)
      magic, num = struct.unpack(">II", f.read(8))
      
      # Citim etichetele (cifre de la 0 la 9)
      y_labels_full_tensor = np.fromfile(f, dtype=np.uint8)

   X_images_full_tensor: np.ndarray = X_images_full_tensor.astype("float32") / 255.0

   return X_images_full_tensor, y_labels_full_tensor 

def separate_training_data(X_tensor, y_tensor, training_data_size: int = 50000):
   X_training_data: np.ndarray = X_tensor[:training_data_size]
   y_training_data: np.ndarray = y_tensor[:training_data_size]

   X_validation_data: np.ndarray = X_tensor[training_data_size:]
   y_validation_data: np.ndarray = y_tensor[training_data_size:]

   return X_training_data, y_training_data, X_validation_data, y_validation_data

def load_images_and_masks_tensors(file, images_path, masks_path, selection=slice(None)):
    # 1. Găsim căile către fișiere
    base = data_path(file)
    image_files = sorted(list((base / images_path).glob("*.png")))[selection]
    mask_files = sorted(list((base / masks_path).glob("*.png")))[selection]

    images, masks = [], []
    
    # 2. Procesăm perechile
    for img_p, msk_p in zip(image_files, mask_files):
        # Imagine RGB: citire -> decode -> resize -> normalizare
        img = tf.io.read_file(str(img_p))
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (128, 128)) / 255.0
        images.append(img)
        
        # Mască Grayscale: citire -> decode -> resize (nearest)
        mask = tf.io.read_file(str(msk_p))
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, (128, 128), method='nearest')
        masks.append(mask)
      
    return tf.stack(images), tf.stack(masks)

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