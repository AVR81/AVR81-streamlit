"""
═══════════════════════════════════════════════════════════════════════════
  APLICACIÓN WEB — Clasificador de flores Iris con Streamlit
═══════════════════════════════════════════════════════════════════════════
  Interfaz web que carga el modelo entrenado (models/iris_model.pkl) y
  predice la especie de una flor Iris a partir de sus 4 medidas.

  Ejecutar en local:
      streamlit run app.py
═══════════════════════════════════════════════════════════════════════════
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Iris",
    page_icon="🌸",
    layout="centered",
)


# ── Carga del modelo (cacheada para no recargar en cada interacción) ─────────
@st.cache_resource
def cargar_modelo():
    ruta = os.path.join(os.path.dirname(__file__), "models", "iris_model.pkl")
    paquete = joblib.load(ruta)
    return paquete["model"], paquete["target_names"], paquete["feature_names"]


model, target_names, feature_names = cargar_modelo()


# ── Cabecera ────────────────────────────────────────────────────────────────
st.title("🌸 Clasificador de flores Iris")
st.markdown(
    "Esta aplicación predice la **especie** de una flor Iris a partir de las "
    "medidas de su sépalo y su pétalo. Ajusta los deslizadores en la barra "
    "lateral y pulsa **Predecir**."
)

st.divider()


# ── Barra lateral: entradas del usuario ─────────────────────────────────────
st.sidebar.header("Medidas de la flor (cm)")

sepal_length = st.sidebar.slider("Longitud del sépalo", 4.0, 8.0, 5.8, 0.1)
sepal_width  = st.sidebar.slider("Anchura del sépalo",   2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Longitud del pétalo",  1.0, 7.0, 4.3, 0.1)
petal_width  = st.sidebar.slider("Anchura del pétalo",   0.1, 2.5, 1.3, 0.1)

# Mostrar un resumen de las medidas seleccionadas
st.subheader("Medidas seleccionadas")
entrada = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=feature_names,
)
st.dataframe(entrada, use_container_width=True, hide_index=True)


# ── Predicción ──────────────────────────────────────────────────────────────
if st.button("🔮 Predecir especie", type="primary", use_container_width=True):
    pred = model.predict(entrada)[0]
    proba = model.predict_proba(entrada)[0]
    especie = target_names[pred]

    st.divider()
    st.success(f"### Especie predicha: **{especie.capitalize()}**")

    # Probabilidades por clase
    st.subheader("Probabilidad por especie")
    proba_df = pd.DataFrame({
        "Especie": [n.capitalize() for n in target_names],
        "Probabilidad": proba,
    }).sort_values("Probabilidad", ascending=False)

    st.bar_chart(proba_df.set_index("Especie"), color="#2ecc71")

    for _, fila in proba_df.iterrows():
        st.write(f"- **{fila['Especie']}**: {fila['Probabilidad']:.1%}")


# ── Pie de página ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Modelo: Random Forest entrenado sobre el dataset Iris (scikit-learn). "
    "Aplicación desarrollada con Streamlit y desplegada en Render."
)
