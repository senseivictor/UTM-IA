import os
import keras
import numpy as np
import pandas as pd

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
csv_test_path: str = os.path.join(BASE_DIR, 'data', 'fashion-mnist_test.csv')
model_save_path = os.path.join(BASE_DIR, 'model', 'model_fashion.h5')

# 1. Încărcarea modelului salvat
print("Se încarcă modelul pre-antrenat...")
model: keras.Sequential = keras.models.load_model(model_save_path) # type: ignore

# 2. Încărcarea datelor de test
print("Se încarcă datele de test...")
test_df: pd.DataFrame = pd.read_csv(csv_test_path)

y_test: np.ndarray = test_df['label'].values
X_test_raw: np.ndarray = test_df.drop('label', axis=1).values.reshape(-1, 28, 28)

# Normalizare (trebuie să fie identică cu cea de la antrenare)
X_test: np.ndarray = X_test_raw.astype("float32") / 255.0

# 3. Evaluarea performanței
print("\nEvaluarea pe setul de date de test:")
results: list[float] = model.evaluate(X_test, y_test, verbose=2) # type: ignore
print(f'\nAcuratețea modelului: {results[1]*100:.2f}%')

# 4. Predicții individuale
class_names: list[str] = ['Tricou', 'Pantaloni', 'Pulover', 'Rochie', 'Palton', 
                          'Sandală', 'Cămașă', 'Adidași', 'Geantă', 'Ghete']

# Luăm primele 5 imagini din setul de test
predictions: np.ndarray = model.predict(X_test[:5])

print("\nRezultate predicții individuale:")
for i in range(5):
    predicted_idx: int = int(np.argmax(predictions[i]))
    actual_idx: int = int(y_test[i])
    print(f"Imaginea {i+1}: Predicție -> {class_names[predicted_idx]} | Real -> {class_names[actual_idx]}")