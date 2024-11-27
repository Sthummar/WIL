# Streamlit App for Voter Prediction
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load the trained model and scaler
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app
st.title("Voter Behavior Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
income = st.selectbox("Income Level", ["Low", "Medium", "High"])
location = st.selectbox("Location", ["Urban", "Rural", "Suburban"])
past_voting_behavior = st.selectbox("Past Voting Behavior", ["Voter", "Non-voter"])
social_media_engagement = st.slider("Social Media Engagement", min_value=0, max_value=100, value=50)
positive_sentiment_posts = st.number_input("Positive Sentiment Posts", min_value=0, value=10)
negative_sentiment_posts = st.number_input("Negative Sentiment Posts", min_value=0, value=5)

# Button to make predictions
if st.button("Predict"):
    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'age': [age],
        'social_media_engagement': [social_media_engagement],
        'positive_sentiment_posts': [positive_sentiment_posts],
        'negative_sentiment_posts': [negative_sentiment_posts],
        'gender_Female': [1 if gender == "Female" else 0],
        'gender_Non-binary': [1 if gender == "Non-binary" else 0],
        'education_Bachelor\'s': [1 if education == "Bachelor's" else 0],
        'education_Master\'s': [1 if education == "Master's" else 0],
        'education_PhD': [1 if education == "PhD" else 0],
        'income_Medium': [1 if income == "Medium" else 0],
        'income_High': [1 if income == "High" else 0],
        'location_Rural': [1 if location == "Rural" else 0],
        'location_Suburban': [1 if location == "Suburban" else 0],
        'past_voting_behavior_Non-voter': [1 if past_voting_behavior == "Non-voter" else 0]
    })

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(scaled_data)
    predicted_label = "Will Vote" if prediction[0] == 1 else "Will Not Vote"

    # Display the prediction
    st.write(f"Prediction: {predicted_label}")
