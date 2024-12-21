import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the pre-trained model 
model = load('model\logistic_regression_model.joblib')

# List of expected columns (the same as when the model was trained)
expected_columns = [
    'Lead_ID', 'Contacted', 'Follow_Ups', 
    'Source_Social Media', 'Source_Referral', 'Source_Website', 
    'Interest_Level_Medium', 'Interest_Level_Low', 
    'Program_Offered_Data Science', 'Program_Offered_Mathematics', 
    'Program_Offered_Physics'
]

# Function to preprocess input data (same as training preprocessing)
def preprocess_input_data(data):
    # Convert input data to DataFrame
    df = pd.DataFrame([data])

    # One-hot encode categorical variables, making sure to drop the first category to avoid multicollinearity
    df_encoded = pd.get_dummies(df, columns=['Source', 'Interest_Level', 'Program_Offered'], drop_first=True)

    # Ensure all columns from the training set are present (add missing columns with value 0)
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder the columns to match the model's expected order
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded

# Streamlit UI
st.title("Lead Conversion Prediction")

# User input form
lead_data = {
    'Lead_ID': st.number_input("Lead ID", min_value=1, step=1),
    'Source': st.selectbox("Source", ['Website', 'Event', 'Social Media', 'Referral']),
    'Interest_Level': st.selectbox("Interest Level", ['High', 'Medium', 'Low']),
    'Contacted': st.number_input("Contacted", min_value=0, step=1),
    'Follow_Ups': st.number_input("Follow Ups", min_value=0, step=1),
    'Program_Offered': st.selectbox("Program Offered", ['Biology', 'Data Science', 'Mathematics', 'Physics']),
}

# Button to make prediction
if st.button('Predict Conversion'):
    # Preprocess input data
    input_data = preprocess_input_data(lead_data)
    
    # Ensure the model gets the data in the correct format (Numpy array)
    input_data = input_data.values

    # Make prediction using the model
    prediction = model.predict(input_data)
    
    # Show result
    if prediction == 1:
        st.success("This lead is likely to convert.")
    else:
        st.error("This lead is unlikely to convert.")
