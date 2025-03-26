import streamlit as st 
import requests
import json 
import pandas as pd
from PIL import Image



st.markdown("""
    <style>
    .custom-container {
        background-color: #f5f5dc; /* Cream color */
        color: black;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)



# header image:
image = "./images/churn.jpg"


col1, col2 = st.columns([4, 6])


with col1:
    st.image(image, width=800)


with col2:
    st.title("Customer Churn Prediction Model")
   


st.write("""
    **This churn prediction model** leverages machine learning to predict 
    customer attrition by analyzing key factors, helping businesses take proactive 
    actions to retain valuable customers.
""")
    
container = st.container()
col1, col2 = container.columns([1, 1])

with col1:
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.write(""" 
        Customer Churn refers to the loss of customers over a specific period. 
        Understanding churn is crucial for businesses as it helps identify at-risk customers, 
        allowing proactive measures to retain them. By predicting churn, companies can improve customer 
        satisfaction, reduce acquisition costs, and enhance overall profitability. Early detection of churn enables businesses 
        to take timely action, such as offering incentives or personalized services, 
        to keep valuable customers and maintain long-term success.
    """)
    st.markdown('</div>', unsafe_allow_html=True)