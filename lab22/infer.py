import os
import keras
import numpy as np
import pandas as pd
from utils import load_ubyte_tensors, get_this_file_dir

class_names: list[str] = ['Tricou', 'Pantaloni', 'Pulover', 'Rochie', 'Palton', 
                          'Sandală', 'Cămașă', 'Adidași', 'Geantă', 'Ghete']

model: keras.Sequential = keras.models.load_model(
    os.path.join(get_this_file_dir(), 'model', 'model.keras')
)

X_tensor, y_tensor = load_ubyte_tensors(
    'test/t10k-images-idx3-ubyte', 
    'test/t10k-labels-idx1-ubyte'
)

# results = [loss, accuracy]
results: list[float, float] = model.evaluate(X_tensor, y_tensor)
print(f'\nAcuratețea modelului: {results[1]*100:.2f}%')

# Luăm primele 5 imagini din setul de test
predictions: np.ndarray = model.predict(X_tensor[:5])

print("\nRezultate predicții individuale:")
for i in range(5):
    predicted_idx: int = int(np.argmax(predictions[i]))
    actual_idx: int = int(y_tensor[i])
    print(f"Imaginea {i+1}: Predicție -> {class_names[predicted_idx]} | Real -> {class_names[actual_idx]}")