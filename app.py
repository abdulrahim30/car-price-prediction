import streamlit as st
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

# Load the XGBoost model from the pickle file
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Car Price Prediction")

# Input fields
year = st.number_input("Year", min_value=1900, max_value=2024, step=1)
make = st.text_input("Make")
model_name = st.text_input("Model")
odometer = st.number_input("Odometer", min_value=0, step=1)
condition = st.number_input("Condition", min_value=1, step=1)

# Button for prediction
if st.button("Predict"):
    # Prepare the input data
    # input_data = pd.DataFrame([[year, make, model_name, odometer, condition]])
    input_data = pd.DataFrame(
        [[year, make, model_name, odometer, condition]],
        columns=["year", "make", "model", "odometer", "condition"],
    )

    # Run the model prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f"Predicted Car Price: ${prediction[0]:,.2f}")
