import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("📈 Stock Market Trend Predictor")

# Ticker input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")

if st.button("Predict Trend"):

    # Load data
    data = yf.download(ticker, start="2015-01-01", end="2024-12-31", auto_adjust=False)
    data.dropna(inplace=True)

    # Feature Engineering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_20', 'EMA_10']
    X = data[features]
    y = data['Target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on last day
    latest_data = X.iloc[[-1]]
    prediction = model.predict(latest_data)[0]

    # Show prediction
    st.success(f"📊 Prediction for next day: {'📈 Up' if prediction == 1 else '📉 Down'}")

    # Show chart
    st.subheader("Closing Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['Close'], label="Close")
    ax.plot(data['SMA_10'], label="SMA 10")
    ax.plot(data['SMA_20'], label="SMA 20")
    ax.legend()
    st.pyplot(fig)
