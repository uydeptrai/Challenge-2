import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Function to load models


def load_models():
    models = {}
    models['my_model'] = load_model('my_model.h5')
    models['my_model_1'] = load_model('my_model_1.h5')
    return models

# Function to make predictions


def make_predictions(model, X_test, scaler):
    y_predicted = model.predict(X_test)
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    return y_predicted


# Customizing Streamlit frontend
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Custom CSS styling
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            color: #21618C;
            font-size: 2.5em;
        }
        .input-box {
            width: 50%;
            margin: auto;
            padding: 10px;
            text-align: center;
            background-color: #F4F6F6;
        }
        .plot {
            width: 80%;
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

start = dt.datetime(2017, 1, 1)
end = dt.datetime(2024, 1, 1)

# Streamlit frontend
st.title('Stock Trend Prediction')
st.markdown("<p class='title'>Stock Trend Prediction</p>",
            unsafe_allow_html=True)

user_input = st.text_input('Enter Stock Ticker', 'TSLA', key='user_input')

# Check if user input is 'yahoo'
if user_input.lower() == 'yahoo':
    st.warning("Please enter a valid stock ticker other than 'yahoo'.")
else:
    # Fetch data using yfinance
    bitcoin = yf.download(user_input, start=start, end=end)
    eth_data = yf.download('ETH-USD', start=start, end=end)

    # Describing Data
    st.subheader('Data from 2014-2023')
    st.write(bitcoin.describe())
    st.write(bitcoin.tail(20))

    # Visualization
    st.subheader('Close Price')

    # Plot for BTC
    st.subheader('BTC Close Price')
    fig_btc = plt.figure(figsize=(12, 6))
    plt.plot(bitcoin['Close'])
    st.pyplot(fig_btc, use_container_width=True)

    # Plot for ETH
    st.subheader('ETH Close Price')
    fig_eth = plt.figure(figsize=(12, 6))
    plt.plot(eth_data['Close'])
    st.pyplot(fig_eth, use_container_width=True)

    # Splitting the dataset into Training and Testing
    data = bitcoin.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Load models
    models = load_models()

    # Select the model based on user input
    selected_model = st.selectbox('Select Model', list(models.keys()))

    # Testing Part
    test_data = scaled_data[training_data_len - 60:, :]

    X_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60: i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Get the selected model
    model = models[selected_model]

    # Making Predictions
    y_predicted = make_predictions(model, X_test, scaler)

    # Final Graph
    st.subheader('Prediction and Original')
    fig2, ax = plt.subplots(figsize=(12, 12))
    ax.plot(y_test, 'b', label='Original Price')
    ax.plot(y_predicted, 'r', label='Predicted Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig2)
