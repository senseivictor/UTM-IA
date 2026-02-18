import os
import pandas as pd
import numpy as np
import struct

def get_this_file_dir():
   return os.path.dirname(os.path.abspath(__file__))

def read_training_data_csv():
   print("Se încarcă datele pentru antrenare...")
   path = os.path.join(get_this_file_dir(), 'data', 'fashion-mnist_train.csv')
   return pd.read_csv(path)


def load_ubyte(images_path: str, labels_path: str):
   full_images_path = os.path.join(get_this_file_dir(), 'data', images_path)
   full_labels_path = os.path.join(get_this_file_dir(), 'data', labels_path)
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

   return X_images_full_tensor, y_labels_full_tensor 