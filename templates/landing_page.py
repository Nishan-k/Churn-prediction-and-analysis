# import streamlit as st 
# import requests
# import json 
# import pandas as pd
# from PIL import Image



# st.markdown("""
#     <style>
#     .custom-container {
#         background-color: #f5f5dc; /* Cream color */
#         color: black;
#         padding: 20px;
#         border-radius: 10px;
#     }
#     </style>
# """, unsafe_allow_html=True)



# # header image:
# image = "../images/churn.jpg"


# col1, col2 = st.columns([4, 6])


# with col1:
#     st.image(image, width=800)


# with col2:
#     st.title("Customer Churn Prediction Model")
   


# st.write("""
#     **This churn prediction model** leverages machine learning to predict 
#     customer attrition by analyzing key factors, helping businesses take proactive 
#     actions to retain valuable customers.
# """)
    
# container = st.container()
# col1, col2 = container.columns([1, 1])

# with col1:
#     st.markdown('<div class="custom-container">', unsafe_allow_html=True)
#     st.write(""" 
#         Customer Churn refers to the loss of customers over a specific period. 
#         Understanding churn is crucial for businesses as it helps identify at-risk customers, 
#         allowing proactive measures to retain them. By predicting churn, companies can improve customer 
#         satisfaction, reduce acquisition costs, and enhance overall profitability. Early detection of churn enables businesses 
#         to take timely action, such as offering incentives or personalized services, 
#         to keep valuable customers and maintain long-term success.
#     """)
#     st.markdown('</div>', unsafe_allow_html=True)


import streamlit as st
import requests

# Define FastAPI endpoint (adjust the URL to where your FastAPI app is running)
FASTAPI_URL = "http://localhost:8000/api/churn-prediction"

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Sidebar navigation for pages
page = st.sidebar.selectbox("Choose a page", ["Home", "Prediction"])

# Home Page
if page == "Home":
    st.title("Customer Churn Prediction Model")
    st.subheader("Creator: Nishan Karki")
    st.write("""
    Customer Churn refers to the loss of customers over a specific period. 
    Understanding churn is crucial for businesses as it helps identify at-risk customers, 
    allowing proactive measures to retain them.
    """)
    st.write("""
    In this app, we predict customer churn using machine learning techniques. 
    The prediction helps businesses take proactive actions to retain valuable customers.
    """)

# Prediction Page
if page == "Prediction":
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
