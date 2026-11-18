
import streamlit as st
import pandas as pd
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="Air Pollution Predictor",
    page_icon="🌍",
    layout="centered"
)

# Load saved model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_files():
    model = joblib.load(os.path.join(BASE_DIR, "saved_models/best_model.joblib"))
    imputer = joblib.load(os.path.join(BASE_DIR, "saved_models/imputer.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "saved_models/scaler.joblib"))

    with open(os.path.join(BASE_DIR, "saved_models/metadata.json"), "r") as f:
        metadata = json.load(f)

    return model, imputer, scaler, metadata

model, imputer, scaler, metadata = load_files()

# Application heading
st.title("🌍 Air Pollution Prediction")
st.write("### Predict daily PM2.5 concentration using Machine Learning")

st.info(
    "Enter the weather and pollution measurements below "
    "to estimate the daily PM2.5 concentration."
)

# Input section
st.subheader("Enter Environmental Parameters")

col1, col2 = st.columns(2)

with col1:
    pm10 = st.number_input(
        "PM10 (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-30.0,
        max_value=60.0,
        value=20.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

with col2:
    wind_speed = st.number_input(
        "Wind Speed (m/s)",
        min_value=0.0,
        max_value=100.0,
        value=2.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0
    )

    previous_pm25 = st.number_input(
        "Previous Day PM2.5 (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0
    )

# Prediction button
if st.button("🔍 Predict PM2.5", use_container_width=True):

    input_values = {
        "PM10": pm10,
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind_Speed": wind_speed,
        "Rainfall": rainfall,
        "Previous_Day_PM25": previous_pm25
    }

    input_df = pd.DataFrame(
        [[input_values[feature] for feature in metadata["features"]]],
        columns=metadata["features"]
    ).astype(float)

    # Convert to numpy array to avoid scikit-learn feature-name mismatch
    # between training and inference environments
    input_array = input_df.values

    # Apply the same preprocessing used during training
    input_imputed = imputer.transform(input_array)

    # Scale input only when the selected model is SVR
    if metadata["requires_scaling"]:
        input_processed = scaler.transform(input_imputed)
    else:
        input_processed = input_imputed

    # Make prediction
    prediction = float(model.predict(input_processed)[0])

    # PM2.5 cannot be negative
    prediction = max(0, prediction)

    st.success("Prediction completed!")

    st.metric(
        label="Predicted Daily PM2.5",
        value=f"{prediction:.2f} µg/m³"
    )

    st.caption(
        "This is a machine-learning estimate, not a real-time air-quality measurement."
    )

# Model information
st.divider()

st.subheader("Model Information")

st.write("**Selected Model:**", metadata["best_model"])
st.write("**Mean Absolute Error (MAE):**", round(metadata["MAE"], 3))
st.write("**R² Score:**", round(metadata["R2"], 3))

with st.expander("About this project"):
    st.write(
        "This application predicts daily PM2.5 concentration using "
        "PM10, temperature, humidity, wind speed, rainfall, "
        "and previous-day PM2.5."
    )
    st.write(
        "The machine-learning models were trained using the "
        "Beijing Multi-Site Air Quality dataset."
    )

st.caption("Air Pollution Prediction Using Machine Learning | Academic Project")
