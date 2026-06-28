# 🌸 Aplicación Web ML con Streamlit — Clasificador de Iris

Aplicación web de Machine Learning que predice la **especie de una flor Iris**
(setosa, versicolor o virginica) a partir de las medidas de su sépalo y pétalo.
El modelo es un **Random Forest** entrenado con scikit-learn, y la interfaz está
desarrollada con **Streamlit** y desplegada en **Render**.

## 🔗 Aplicación desplegada

👉 **[PEGAR AQUÍ EL ENLACE DE RENDER UNA VEZ DESPLEGADA]**

> ⚠️ El servicio usa el plan gratuito de Render. Si lleva más de 15 minutos sin
> uso, la primera carga puede tardar entre 30 y 50 segundos en "despertar". Es
> un comportamiento normal del plan free.

---

## 📁 Estructura del proyecto

```
.
├── train_model.py      # Entrena el modelo y genera models/iris_model.pkl
├── app.py              # Aplicación web Streamlit
├── models/
│   └── iris_model.pkl  # Modelo entrenado (Random Forest)
├── requirements.txt    # Dependencias con versiones fijadas
├── render.yaml         # Configuración del servicio en Render
├── .gitignore
└── README.md
```

---

## 🚀 Cómo ejecutar en local

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Opcional) Reentrenar el modelo:**
   ```bash
   python train_model.py
   ```
   > Importante: ejecuta este paso si cambias las versiones de las librerías,
   > para que el `.pkl` sea compatible con la versión de scikit-learn instalada.

3. **Lanzar la aplicación:**
   ```bash
   streamlit run app.py
   ```
   Se abrirá automáticamente en `http://localhost:8501`.

---

## ☁️ Cómo desplegar en Render

1. Sube este repositorio a GitHub.
2. Entra en [render.com](https://render.com) y crea una cuenta gratuita.
3. Pulsa **New +** → **Web Service** y conecta tu repositorio de GitHub.
4. Configura el servicio:
   - **Runtime / Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**:
     ```
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - **Instance Type**: `Free`
5. Pulsa **Create Web Service** y espera a que termine el build.
6. Copia la URL pública que te da Render y pégala en la sección
   "Aplicación desplegada" de este README.

> El archivo `render.yaml` ya incluye esta configuración, por lo que Render
> puede detectarla automáticamente al conectar el repositorio.

---

## 🧠 Sobre el modelo

- **Dataset**: Iris (150 muestras, 4 variables, 3 especies) — incluido en scikit-learn.
- **Algoritmo**: Random Forest (100 árboles).
- **Variables de entrada**: longitud y anchura del sépalo y del pétalo (en cm).
- **Salida**: especie predicha + probabilidad de cada clase.

---

## 📚 Recursos externos utilizados

- [Documentación oficial de Streamlit](https://docs.streamlit.io/)
- [Guía de despliegue de Render](https://render.com/docs/deploy-streamlit)
- [Dataset Iris en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
