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

# Initialize the scaler
scaler = StandardScaler()

# Streamlit app
st.title("Voter Behavior Prediction")

# Streamlit input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
social_media_engagement = st.number_input("Social Media Engagement", min_value=0, value=50)
positive_sentiment_posts = st.number_input("Positive Sentiment Posts", min_value=0, value=10)
negative_sentiment_posts = st.number_input("Negative Sentiment Posts", min_value=0, value=5)

gender = st.selectbox("Gender", options=["Male", "Female", "Non-binary"])
education = st.selectbox("Education Level", options=["High School", "Bachelor's", "Master's", "PhD"])
income = st.selectbox("Income Level", options=["Low", "Medium", "High"])
location = st.selectbox("Location", options=["Urban", "Suburban", "Rural"])
past_voting_behavior = st.selectbox(
    "Past Voting Behavior",
    options=["Non-voter", "Occasional Voter", "Regular Voter"]
)

input_dict = {
    "age": [age],
    "social_media_engagement": [social_media_engagement],
    "positive_sentiment_posts": [positive_sentiment_posts],
    "negative_sentiment_posts": [negative_sentiment_posts],
    "gender_Male": [1 if gender == "Male" else 0],
    "gender_Non-binary": [1 if gender == "Non-binary" else 0],
    "education_High School": [1 if education == "High School" else 0],
    "education_Master's": [1 if education == "Master's" else 0],
    "education_PhD": [1 if education == "PhD" else 0],
    "income_Low": [1 if income == "Low" else 0],
    "income_Medium": [1 if income == "Medium" else 0],
    "location_Suburban": [1 if location == "Suburban" else 0],
    "location_Urban": [1 if location == "Urban" else 0],
    "past_voting_behavior_Non-voter": [1 if past_voting_behavior == "Non-voter" else 0],
    "past_voting_behavior_Occasional Voter": [1 if past_voting_behavior == "Occasional Voter" else 0],
}

input_data = pd.DataFrame(input_dict)


# Scale the input data
scaled_data = scaler.transform(input_data)

# Make predictions
prediction = model.predict(scaled_data)
predicted_label = "Will Vote" if prediction[0] == 1 else "Will Not Vote"

# Display the prediction
st.write(f"Prediction: {predicted_label}")
