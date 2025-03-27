import streamlit as st 
import requests
import json 
import pandas as pd
from PIL import Image
from  customer_churn_ml.data_loader import churn_count
import matplotlib.pyplot as plt



churn_data = churn_count()


# Define FastAPI endpoint (adjust the URL to where your FastAPI app is running)
FASTAPI_URL = "http://localhost:8000/api/churn-prediction"

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Sidebar navigation for pages
page = st.sidebar.selectbox("Choose a page", ["Home", "Predict", "Explain", "Recommendations", "About"])
image = "./images/churn.jpg"
# Home Page
if page == "Home":
     
     st.title("Customer Churn Prediction Model")
        
     st.write("""
        Customer Churn refers to the loss of customers over a specific period. 
        Understanding churn is crucial for businesses as it helps identify at-risk customers, 
        allowing proactive measures to retain them.
        """)
     
     st.subheader("Current Customer Distribution")
     col1, col2 = st.columns([9, 5])
     with col1:        
        plt.figure(figsize=(6, 4))
        fig, ax = plt.subplots()
        ax.bar(churn_data["churn"], churn_data["count"], color=['green', 'red'])
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Count")
        ax.set_title("Customer Churn Distribution")
        st.pyplot(fig)

     with col2:
         st.write(churn_data)
    


# Prediction Page
if page == "Predict":
    st.title("Churn Prediction")

    # User input fields
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.number_input("Tenure (in months)", min_value=1, max_value=72, step=1)
    monthly_spend = st.number_input("Monthly Spend", min_value=0.0, max_value=10000.0, step=0.1)
    
    if st.button("Predict Churn"):
        # Prepare data for API request
        payload = {
            "age": age,
            "contract_type": contract_type,
            "tenure": tenure,
            "monthly_spend": monthly_spend
        }
        
        # Send data to FastAPI for prediction
        response = requests.post(FASTAPI_URL, json=payload)
        
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.write(f"Churn Prediction: {prediction}")
        else:
            st.error("Error in prediction. Please try again.")
