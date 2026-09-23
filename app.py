import os
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "house_price_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "scaler.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "model",
    "feature_names.pkl"
)

try:
    model = load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(FEATURE_PATH, "rb") as f:
        feature_names = pickle.load(f)

    st.success("✅ Model files loaded successfully!")

except Exception as e:

    st.error("❌ Model files could not be loaded.")

    st.code(str(e))

    st.write("Model path:", MODEL_PATH)
    st.write("Scaler path:", SCALER_PATH)
    st.write("Feature path:", FEATURE_PATH)

    st.stop()
