
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# === Load your trained model ===
model = joblib.load("ridge_weather_model.pkl")

# === Streamlit App UI ===
st.title("🌤️ Temperature Forecasting App")
st.markdown("Enter current weather details to predict the **temperature** using Machine Learning.")

# === Input fields ===
humidity = st.slider("Humidity (%)", 0, 100, 65) / 100
pressure = st.number_input("Pressure (millibars)", value=1012.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=5.0)
summary = st.selectbox("Weather Summary", [
    "Clear", "Partly Cloudy", "Mostly Cloudy", "Overcast", "Rain", "Foggy", "Drizzle", "Breezy"
])

# Date input
date_input = st.date_input("Forecast Date", datetime.today())
month = date_input.month
day_of_year = date_input.timetuple().tm_yday
season = (month % 12) // 3 + 1

# Lagged temperature (yesterday's value)
temp_lag1 = st.number_input("Yesterday's Temperature (°C)", value=20.0)

# Derived interaction features
humidity_pressure = humidity * pressure
humidity_wind = humidity * wind_speed

# === Create DataFrame for prediction ===
input_df = pd.DataFrame([{
    "Humidity": humidity,
    "Pressure": pressure,
    "WindSpeed": wind_speed,
    "Summary": summary,
    "Month": month,
    "DayOfYear": day_of_year,
    "Season": season,
    "Temp_lag1": temp_lag1,
    "Humidity_Pressure": humidity_pressure,
    "Humidity_Wind": humidity_wind
}])

# === Make Prediction ===
if st.button("Predict Temperature"):
    predicted_temp = model.predict(input_df)[0]
    st.success(f"🌡️ Predicted Temperature: **{predicted_temp:.2f}°C**")
