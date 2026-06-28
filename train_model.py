"""
═══════════════════════════════════════════════════════════════════════════
  ENTRENAMIENTO DEL MODELO — Clasificador de flores Iris
═══════════════════════════════════════════════════════════════════════════
  Este script carga el dataset Iris, entrena un modelo de clasificación
  (Random Forest) y lo guarda en models/iris_model.pkl con joblib.

  Ejecutar una sola vez en local:
      python train_model.py
═══════════════════════════════════════════════════════════════════════════
"""

import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ── 1. Carga del dataset ────────────────────────────────────────────────────
print("Cargando el dataset Iris...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print(f"  Muestras : {X.shape[0]}")
print(f"  Variables: {list(X.columns)}")
print(f"  Clases   : {list(iris.target_names)}")


# ── 2. División train / test ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")


# ── 3. Entrenamiento del modelo ─────────────────────────────────────────────
print("\nEntrenando Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# ── 4. Evaluación ───────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nPrecisión en test: {acc:.2%}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# ── 5. Guardado del modelo ──────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
ruta_modelo = "models/iris_model.pkl"

# Guardamos el modelo junto con los nombres de clases y variables,
# así la app no depende de volver a importar el dataset.
paquete = {
    "model": model,
    "target_names": list(iris.target_names),
    "feature_names": list(X.columns),
}
joblib.dump(paquete, ruta_modelo)

print(f"\n✓ Modelo guardado en {ruta_modelo}")
print("  Listo para usar en la aplicación Streamlit.")
