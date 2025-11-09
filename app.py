import streamlit as st
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import shap
import json
import os
from dotenv import load_dotenv
load_dotenv() 
# -----------------------------
# ðŸ§  Load trained model and scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("/kaggle/input/indian-oil-aqi-model-files/lstm_aqi_model.h5", compile=False)
    return model

model = load_model()

try:
    scaler = joblib.load("/kaggle/input/indian-oil-aqi-model-files/lstm_scaler.save")
except:
    scaler = None

# -----------------------------
# ðŸ”‘ AQICN API setup
# -----------------------------
# API_TOKEN = os.getenv("AQICN_TOKEN", "e499e8ff9754b7044f92675b0d6d34b1cbfc9025")
API_TOKEN = os.getenv("AQICN_TOKEN")
if not API_TOKEN:
    st.error("API token missing — set AQICN_TOKEN in environment or Streamlit secrets.")

def fetch_live_aqi(city):
    """Fetch live AQI data from AQICN API for the given city."""
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["status"] == "ok":
            iaqi = data["data"]["iaqi"]
            return {
                "City": city,
                "Time": data["data"]["time"]["s"],
                "AQI": data["data"].get("aqi"),
                "PM2.5": iaqi.get("pm25", {}).get("v"),
                "PM10": iaqi.get("pm10", {}).get("v"),
                "NO2": iaqi.get("no2", {}).get("v"),
                "SO2": iaqi.get("so2", {}).get("v"),
                "CO": iaqi.get("co", {}).get("v"),
                "O3": iaqi.get("o3", {}).get("v")
            }
    except Exception as e:
        print(f"Error fetching AQI data: {e}")
    return None

# -----------------------------
# ðŸŽ¨ Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="Indian Oil AQI Dashboard", page_icon="ðŸŒ", layout="wide")

st.title("ðŸŒ Indian Oil AQI Forecast Dashboard")
st.markdown("### Real-time AQI Prediction using LSTM, SHAP, and API Integration")

# City dropdown
cities = [
    "Delhi", "Mumbai", "Chennai", "Kolkata", "Jaipur", "Hyderabad", "Bhopal",
    "Patna", "Lucknow", "Chandigarh", "Visakhapatnam", "Thiruvananthapuram"
]

selected_city = st.selectbox("Select a city to analyze:", cities)

# -----------------------------
# ðŸš€ Fetch and predict
# -----------------------------
if st.button("Fetch Live AQI"):
    with st.spinner("Fetching live AQI data..."):
        data = fetch_live_aqi(selected_city)
        if data:
            df = pd.DataFrame([data])
            st.success(f"âœ… Live AQI data fetched for {selected_city}")
            st.dataframe(df)

            # Input processing
            FEATURES = ['AQI','AQI_lag1','AQI_lag7','AQI_roll_3','AQI_roll_7',
                        'PM2.5','PM10','NO2','SO2','O3','CO']

            vals = [data["AQI"], data["AQI"], data["AQI"], data["AQI"], data["AQI"],
                    data["PM2.5"], data["PM10"], data["NO2"], data["SO2"], data["O3"], data["CO"]]
            input_array = np.array([vals])

            if scaler:
                scaled = scaler.transform(input_array)
            else:
                scaled = input_array / np.nanmax(input_array)

            X_input = np.repeat(scaled[:, np.newaxis, :], 14, axis=1)
            pred = model.predict(X_input, verbose=0)[0][0]
            st.metric(label="ðŸŒ¤ï¸ Predicted Next-Day AQI", value=f"{pred:.2f}")

            # ðŸ“Š Pollutant Visualization
            fig = px.bar(
                x=["PM2.5","PM10","NO2","SO2","CO","O3"],
                y=[data["PM2.5"],data["PM10"],data["NO2"],data["SO2"],data["CO"],data["O3"]],
                labels={"x": "Pollutant", "y": "Concentration (Âµg/mÂ³)"},
                title=f"Pollutant Concentration in {selected_city}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ðŸ’¾ Save output
            df["Predicted_AQI"] = np.round(pred, 2)
            csv_path = f"{selected_city}_Live_AQI_Prediction.csv"
            df.to_csv(csv_path, index=False)
            st.download_button("ðŸ“¥ Download Prediction CSV", data=df.to_csv(index=False), file_name=csv_path)

            # ðŸ“˜ SHAP explainability
            with st.expander("ðŸ” Model Explainability (SHAP)"):
                try:
                    background = np.repeat(scaled[np.newaxis, :, :], 10, axis=0)
                    explainer = shap.DeepExplainer(model, background)
                    shap_values = explainer.shap_values(X_input)
                    shap_df = pd.DataFrame(shap_values[0][0], columns=FEATURES)
                    shap_summary = shap_df.mean().sort_values(ascending=False)
                    st.bar_chart(shap_summary)
                except Exception as e:
                    st.warning(f"SHAP could not be computed here due to runtime limits: {e}")

        else:
            st.error("âš ï¸ Failed to fetch AQI data. Try again later.")

# -----------------------------
# ðŸ“œ Sidebar and footer
# -----------------------------
st.sidebar.header("ðŸ“Š Model Information")
st.sidebar.json({
    "Model": "LSTM AQI Forecaster",
    "Framework": "TensorFlow / Keras",
    "Explainability": "SHAP",
    "Optimization": "TFLite Quantization",
    "Data Source": "AQICN API & CPCB Dataset",
    "Developer": "Enactus R&D (Gautam_1412)"
})

st.markdown("---")
st.caption("Developed by Enactus R&D â€¢ AIML for Environment â€¢ Â© 2025")
