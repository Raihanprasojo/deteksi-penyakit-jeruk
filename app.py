import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# LOAD MODEL (FIX KERAS 3)
# =========================
@st.cache_resource
def load_my_model():
    model = tf.keras.layers.TFSMLayer(
        "model_saved",
        call_endpoint="serving_default"
    )
    return model

model = load_my_model()

# =========================
# CLASS
# =========================
class_names = ['citrus_canker', 'healthy', 'melanose']

# =========================
# UI
# =========================
st.title("🍊 Deteksi Penyakit Daun Jeruk")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

# =========================
# PREDICT
# =========================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # 🔥 PREDICT (BEDA!)
    pred = model(img)
    pred = pred.numpy()

    st.write(class_names[np.argmax(pred)])