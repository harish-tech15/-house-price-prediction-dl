import os
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

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


@st.cache_resource
def load_prediction_model():

    model = load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    with open(FEATURE_PATH, "rb") as file:
        feature_names = pickle.load(file)

    return model, scaler, feature_names


# Load files
try:
    model, scaler, feature_names = load_prediction_model()

except Exception as e:

    st.error("❌ Model files could not be loaded.")

    st.write("Please check that your GitHub repository contains:")

    st.code(
        """
model/
├── house_price_model.keras
├── scaler.pkl
└── feature_names.pkl
        """
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🏠 House Price Prediction")

st.markdown(
    """
### Deep Learning House Price Predictor

Enter the house and neighborhood details below and our
**Artificial Neural Network (ANN)** will estimate the
median house value.
"""
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("🏡 Enter House Details")


# California Housing Dataset features

MedInc = st.number_input(
    "💰 Median Income",
    min_value=0.0,
    max_value=20.0,
    value=3.5,
    step=0.1,
    help="Median income of the area in tens of thousands of dollars."
)

HouseAge = st.number_input(
    "🏚️ House Age",
    min_value=1.0,
    max_value=100.0,
    value=30.0,
    step=1.0,
    help="Median age of houses in the area."
)

AveRooms = st.number_input(
    "🛏️ Average Rooms",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.1,
    help="Average number of rooms per household."
)

AveBedrms = st.number_input(
    "🛌 Average Bedrooms",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Average number of bedrooms per household."
)

Population = st.number_input(
    "👨‍👩‍👧 Population",
    min_value=1.0,
    max_value=50000.0,
    value=1000.0,
    step=100.0,
    help="Population of the block."
)

AveOccup = st.number_input(
    "👥 Average Occupancy",
    min_value=0.5,
    max_value=20.0,
    value=3.0,
    step=0.1,
    help="Average number of people per household."
)

Latitude = st.number_input(
    "🌎 Latitude",
    min_value=32.0,
    max_value=42.0,
    value=35.0,
    step=0.01
)

Longitude = st.number_input(
    "🌎 Longitude",
    min_value=-125.0,
    max_value=-114.0,
    value=-119.0,
    step=0.01
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
)


if predict_button:

    try:

        # Create input DataFrame in EXACT training feature order
        input_data = np.array([[
            MedInc,
            HouseAge,
            AveRooms,
            AveBedrms,
            Population,
            AveOccup,
            Latitude,
            Longitude
        ]])

        # Scale input using trained scaler
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(
            input_scaled,
            verbose=0
        )

        # Model output is in $100,000 units
        predicted_value = float(prediction[0][0])

        # Prevent negative prediction
        predicted_value = max(predicted_value, 0)

        # Convert to dollars
        price_usd = predicted_value * 100000

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success("✅ Prediction completed successfully!")

        st.subheader("🏠 Estimated House Price")

        st.metric(
            label="Predicted Median House Value",
            value=f"${price_usd:,.2f}"
        )

        st.info(
            f"""
            **Estimated Value:** ${price_usd:,.2f}

            The prediction was generated using a
            Deep Learning Artificial Neural Network (ANN).
            """
        )

        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.divider()

        st.subheader("📋 Input Summary")

        col1, col2 = st.columns(2)

        with col1:

            st.write(f"**Median Income:** {MedInc:.2f}")
            st.write(f"**House Age:** {HouseAge:.0f} years")
            st.write(f"**Average Rooms:** {AveRooms:.2f}")
            st.write(f"**Average Bedrooms:** {AveBedrms:.2f}")

        with col2:

            st.write(f"**Population:** {Population:,.0f}")
            st.write(f"**Average Occupancy:** {AveOccup:.2f}")
            st.write(f"**Latitude:** {Latitude:.4f}")
            st.write(f"**Longitude:** {Longitude:.4f}")

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.write("Error details:")

        st.code(str(e))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 About the Model")

    st.write(
        """
        This application uses a Deep Learning
        Artificial Neural Network to predict
        median house values.
        """
    )

    st.divider()

    st.write("### 🧠 Model")

    st.write(
        """
        - TensorFlow
        - Keras
        - Artificial Neural Network
        - StandardScaler
        """
    )

    st.divider()

    st.write("### 📊 Features")

    for feature in feature_names:

        st.write(f"• {feature}")

    st.divider()

    st.caption(
        "House Price Prediction using Deep Learning"
    )