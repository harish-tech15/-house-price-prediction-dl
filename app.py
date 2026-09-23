import os
import numpy as np
import streamlit as st
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("🏠 House Price Prediction")
st.write("Deep Learning based House Price Prediction")

st.divider()

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")

except Exception as e:
    st.error("❌ Model loading failed")
    st.code(str(e))
    st.stop()

# -----------------------------
# LOAD SCALER
# -----------------------------
try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(FEATURE_PATH, "rb") as f:
        feature_names = pickle.load(f)

except Exception as e:
    st.error("❌ Scaler or feature files loading failed")
    st.code(str(e))
    st.stop()

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("🏡 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=5.0
    )

    house_age = st.number_input(
        "House Age",
        min_value=0.0,
        value=20.0
    )

    ave_rooms = st.number_input(
        "Average Rooms",
        min_value=0.0,
        value=5.0
    )

    ave_bedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.0,
        value=1.0
    )

with col2:
    population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000.0
    )

    ave_occup = st.number_input(
        "Average Occupancy",
        min_value=0.0,
        value=3.0
    )

    latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.0
    )

# -----------------------------
# PREDICTION
# -----------------------------
st.divider()

if st.button("🔮 Predict House Price", use_container_width=True):

    input_data = np.array([[
        med_inc,
        house_age,
        ave_rooms,
        ave_bedrms,
        population,
        ave_occup,
        latitude,
        longitude
    ]])

    try:
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled, verbose=0)

        price = float(prediction[0][0]) * 100000

        if price < 0:
            price = 0

        st.success("🎉 Prediction Completed!")

        st.metric(
            label="🏠 Predicted House Price",
            value=f"${price:,.2f}"
        )

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "Built using Python, TensorFlow, Keras, Deep Learning and Streamlit"
)
