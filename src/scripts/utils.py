import sys
import os
import pandas as pd
import numpy as np
import struct
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

def get_this_file_dir():
   return os.path.dirname(os.path.abspath(__file__))

def load_ubyte_tensors(images_path: str, labels_path: str):
   full_images_path = images_path
   full_labels_path = labels_path
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

def load_images_and_masks_tensors(images_path, masks_path):
   import glob

   image_files = sorted(glob.glob(os.path.join(images_path, "*.png")))
   mask_files = sorted(glob.glob(os.path.join(masks_path, "*.png")))
   
   images = []
   masks = []
   
   print(f"Incarcam {len(image_files)} imagini din {images_path}...")
   
   for img_p, msk_p in zip(image_files, mask_files):
      img = tf.io.read_file(img_p)
      img = tf.image.decode_png(img, channels=3)
      img = tf.image.resize(img, (128, 128))
      img = img / 255.0
      images.append(img)
      
      mask = tf.io.read_file(msk_p)
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