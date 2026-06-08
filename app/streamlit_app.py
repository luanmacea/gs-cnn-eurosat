"""
Demonstração funcional — Streamlit.

Carrega o melhor modelo (CNN-B) e classifica imagens de satélite enviadas pelo
usuário, exibindo a classe predita, a confiança e a distribuição de
probabilidades entre as 10 classes.

Como rodar (a partir da raiz do projeto):
    streamlit run app/streamlit_app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Permite importar de src/ independentemente de onde o streamlit foi chamado
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import CLASS_NAMES, IMG_SIZE  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "cnn_b_best.keras")

# Rótulos em português para a interface
CLASS_PT = {
    "AnnualCrop": "Cultura anual",
    "Forest": "Floresta",
    "HerbaceousVegetation": "Vegetação herbácea",
    "Highway": "Rodovia",
    "Industrial": "Área industrial",
    "Pasture": "Pastagem",
    "PermanentCrop": "Cultura permanente",
    "Residential": "Área residencial",
    "River": "Rio",
    "SeaLake": "Mar / Lago",
}

st.set_page_config(
    page_title="Classificação de Uso do Solo — EuroSAT",
    page_icon="🛰️",
    layout="centered",
)


@st.cache_resource
def get_model():
    """Carrega o modelo uma única vez (cacheado entre interações)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    """Converte a imagem para o formato esperado pelo modelo.

    O modelo já contém a camada Rescaling(1/255), então só precisamos
    redimensionar para 64x64 e entregar pixels uint8 [0,255].
    """
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype="uint8")
    return np.expand_dims(arr, axis=0)


st.title("🛰️ Classificação de Uso e Cobertura do Solo")
st.caption(
    "CNN treinada do zero sobre imagens do satélite Sentinel-2 (dataset EuroSAT "
    "RGB). Componente de visão computacional de uma plataforma de agregação de "
    "dados meteorológicos e de sensoriamento remoto."
)

model = get_model()
if model is None:
    st.error(
        "Modelo não encontrado em `models/cnn_b_best.keras`. "
        "Treine a CNN-B (notebook `03_treino_cnn_b.ipynb`) antes de rodar a demo."
    )
    st.stop()

uploaded = st.file_uploader(
    "Envie uma imagem de satélite (PNG/JPG)", type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagem enviada", use_column_width=True)

    proba = model.predict(preprocess(image), verbose=0)[0]
    top = int(proba.argmax())

    with col2:
        st.metric(
            "Classe predita",
            CLASS_PT[CLASS_NAMES[top]],
            f"{proba[top] * 100:.1f}% de confiança",
        )

    st.subheader("Probabilidades por classe")
    ordem = np.argsort(proba)[::-1]
    df = pd.DataFrame(
        {"probabilidade": [float(proba[i]) for i in ordem]},
        index=[CLASS_PT[CLASS_NAMES[i]] for i in ordem],
    )
    st.bar_chart(df)

    st.info(
        "Numa operação real, esta classe alimentaria a camada de uso do solo da "
        "plataforma, que cruza o resultado com séries meteorológicas "
        "(precipitação, temperatura, índices de vegetação) da mesma coordenada "
        "geográfica — dando contexto agronômico aos dados que o produtor já paga."
    )
else:
    st.write(
        "Aguardando o upload de uma imagem. Dica: use uma amostra do próprio "
        "conjunto de teste do EuroSAT para a demonstração."
    )
