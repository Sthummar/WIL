import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the scaler
scaler = StandardScaler()

# Streamlit interface
st.title("Voter Prediction App")
st.write("Enter the features to predict the vote intention.")

# Create input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.selectbox("Income", options=["Low", "Medium", "High"], index=1)

income_mapping = {"Low": 0, "Medium": 1, "High": 2}
income_value = income_mapping[income]

social_media_engagement = st.number_input("Social Media Engagement", min_value=0, value=50)
positive_sentiment_posts = st.number_input("Positive Sentiment Posts", min_value=0, value=10)
negative_sentiment_posts = st.number_input("Negative Sentiment Posts", min_value=0, value=5)

# Prepare input data for prediction
features = np.array([[age, income_value, social_media_engagement, positive_sentiment_posts, negative_sentiment_posts]])

# Scale the features
features_scaled = scaler.fit_transform(features)

# Make prediction
if st.button("Predict Vote Intention"):
    prediction = model.predict(features_scaled)
    vote_intention = "Vote Yes" if prediction[0] == 1 else "Vote No"
    st.write(f"Prediction: {vote_intention}")
