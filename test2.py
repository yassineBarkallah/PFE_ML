import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the Random Forest model with joblib
with open('random_forest_model.pkl', 'rb') as file:
    model = joblib.load(file)

# Load the scaler with joblib
with open('scaler.pkl', 'rb') as file:
    scaler = joblib.load(file)

# Add CSS styles
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
        padding: 20px;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #4CAF50;
        border: none;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        text-align: center;
        color: #2E86C1;
    }
    .stTitle {
        color: #1F618D;
        font-weight: bold;
        font-size: 24px;
    }
    .stSubheader {
        color: #34495E;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<h1 class='stHeader'>Anticipation des Performances des Machines</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='stHeader'>SAGEMCOM</h2>", unsafe_allow_html=True)

# User inputs for selected features
st.markdown("<h3 class='stSubheader'>Entrées utilisateur</h3>", unsafe_allow_html=True)
air_temperature = st.number_input("Air Temperature [C]:", min_value=-50.0, max_value=100.0, value=0.0, step=0.1)
tool_wear = st.number_input("Tool wear [min]:", min_value=0.0, value=0.0, step=0.1)
rotational_speed = st.number_input("Rotational speed [rpm]:", min_value=0.0, value=0.0, step=0.1)

# Prediction
if st.button("Predict"):
    # Prepare the features
    final_features = pd.DataFrame({
        'Air temperature [C]': [air_temperature],
        'Tool wear [min]': [tool_wear],
        'Rotational speed [rpm]': [rotational_speed]
    })
    
    # Scale the features
    final_features_scaled = scaler.transform(final_features)
    
    # Make the prediction
    prediction = model.predict(final_features_scaled)
    
    # Display the original results
    st.markdown("<h3 class='stSubheader'>Résultats de la Prédiction</h3>", unsafe_allow_html=True)
    st.write(f"Air Temperature: {air_temperature} °C")
    st.write(f"Tool wear: {tool_wear} min")
    st.write(f"Rotational speed: {rotational_speed} rpm")
    st.write(f"Prediction: {prediction[0]}")  # Display the original result

    # Display a table of the features and prediction
    result_df = pd.DataFrame({
        'AIR TEMPERATURE [C]': [air_temperature],
        'TOOL WEAR [min]': [tool_wear],
        'ROTATIONAL SPEED [rpm]': [rotational_speed],
        'PREDICTION': [prediction[0]]
    })
    
    st.write(result_df)
