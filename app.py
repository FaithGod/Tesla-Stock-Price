import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Tesla Stock Prediction", layout="centered")

st.title("🚗 Tesla Stock Price Prediction (LSTM)")
st.write("Predict future Tesla stock closing prices using a trained LSTM model.")

# Load model (modern Keras format – no compile issues)
model = load_model("lstm_final_tuned_model.keras")

forecast_days = st.selectbox("Select Forecast Horizon (Days)", [1, 5, 10])

uploaded_file = st.file_uploader("Upload TSLA.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    data = df[["Adj Close"]]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    window_size = 60
    last_window = scaled_data[-window_size:]
    X_input = last_window.reshape(1, window_size, 1)

    prediction = model.predict(X_input)
    prediction = prediction[:, :forecast_days]

    predicted_prices = scaler.inverse_transform(prediction)[0]

    st.subheader("📈 Predicted Prices")
    for i, price in enumerate(predicted_prices, 1):
        st.write(f"Day {i}: {price:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(predicted_prices, marker="o")
    plt.title("Future Stock Price Prediction")
    plt.xlabel("Days Ahead")
    plt.ylabel("Price")
    st.pyplot(plt)
