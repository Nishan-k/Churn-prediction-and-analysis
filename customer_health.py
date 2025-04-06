import streamlit as st
from customer_churn_ml.data_loader import get_churn_distribution, get_churn_count, get_total_customer_counts
import pandas as pd
from customer_churn_ml.shap import create_clean_shap_dashboard

total_churn_count = get_churn_count()
total_customers = get_total_customer_counts()
baseline_churn_rate = (total_churn_count / total_customers) * 100

contract_mapping = {
    "Month-to-month": 6,
    "One year": 12,
    "Two year": 24
}

# def customer_health_dashboard(res, input_features):
#     st.session_state.res = res
#     st.session_state.input_features = input_features
#     prediction = res.json()['Prediction']
#     st.session_state.prediction_result = prediction
#     prediction_prob = res.json()['Prediction_proba'] * 100
#     delta_precentage = abs(baseline_churn_rate  - prediction_prob)


#     st.write("")
#     st.subheader("Customer Health Dashboard")
    
#     m1, m2, m3 = st.columns(3)

#     # Delta percentage:
#     m1.metric("Churn Risk", 
#                 "🟢 Low" if prediction == 0 else "🔴 High",
#                 delta=f"{delta_precentage:.2f}% better than average" if prediction == 0 else f"{delta_precentage:.2f}% worse than average")
    
#     # Prediction Probability:
#     m2.metric(label= "Prediction Confidence", value=f"{prediction_prob:.2f}%", delta="Model's confidence")

#     # Customer Life-Time Value:
#     contract_length = contract_mapping.get(input_features["contract"])
#     expected_remaining_tenure = max(contract_length - input_features["tenure"], 0)
#     ltv = input_features["monthly_charges"] * expected_remaining_tenure

#     m3.metric("Customer Life Time Value", f"{ltv:.2f} €", delta="Expected Amount")

#     # Risk visualization
#     risk_level = 100 - prediction_prob if prediction == 0 else prediction_prob
#     st.write("Risk Level")
#     st.progress(int(risk_level))

#     st.write("")
    
    
#     customer_data = pd.DataFrame.from_dict({k: [v] for k, v in input_features.items()})
#     result = create_clean_shap_dashboard(customer_data=customer_data)
#     shap_values = result["shap_values"]
#     st.session_state.customer_data = customer_data
#     st.session_state.shap_values = shap_values

#     st.subheader("Prediction Result")
#     prediction = result["prediction"]
#     probability = result["churn_probability"] * 100

#     # Display prediction with formatting
#     if prediction == "Churn":
#         st.error(f"Customer is predicted to churn with {probability:.1f}% probability")
#     else:
#         st.success(f"Customer is predicted to stay with {(100-probability):.1f}% probability")
    
#     # Display the plot
#     st.subheader("Feature Impact Analysis")
#     st.pyplot(result["plot"])
#     st.write("")
#     st.write("")
#     st.info("👉Now, you can go to '📖 Explain' page or '💡 Recommendations' or 📑 Generate Report page for further actions for this customer in the Navigation bar.")
#     st.write("")








# ###------------------------------------------




def customer_health_dashboard(res, input_features):
    # Store all necessary data in session state
    st.session_state.res = res
    st.session_state.input_features = input_features
    st.session_state.dashboard_displayed = True  # Flag to indicate dashboard has been displayed
    
    # Extract and store prediction data
    prediction = res.json()['Prediction']
    st.session_state.prediction_result = prediction
    prediction_prob = res.json()['Prediction_proba'] * 100
    delta_precentage = abs(baseline_churn_rate - prediction_prob)
    
    # Store these values in session state too
    st.session_state.prediction_prob = prediction_prob
    st.session_state.delta_percentage = delta_precentage
    
    # Create and display the dashboard
    display_dashboard()
    
def display_dashboard():
    """Function to display the dashboard using data from session state"""
    if not st.session_state.get('dashboard_displayed', False):
        return
    
    # Get values from session state
    prediction = st.session_state.prediction_result
    prediction_prob = st.session_state.prediction_prob
    delta_precentage = st.session_state.delta_percentage
    input_features = st.session_state.input_features

    st.write("")
    st.subheader("Customer Health Dashboard")
    
    m1, m2, m3 = st.columns(3)

    # Delta percentage:
    m1.metric("Churn Risk", 
            "🟢 Low" if prediction == 0 else "🔴 High",
            delta=f"{delta_precentage:.2f}% better than average" if prediction == 0 else f"{delta_precentage:.2f}% worse than average")
    
    # Prediction Probability:
    m2.metric(label= "Prediction Confidence", value=f"{prediction_prob:.2f}%", delta="Model's confidence")

    # Customer Life-Time Value:
    contract_length = contract_mapping.get(input_features["contract"])
    expected_remaining_tenure = max(contract_length - input_features["tenure"], 0)
    ltv = input_features["monthly_charges"] * expected_remaining_tenure

    m3.metric("Customer Life Time Value", f"{ltv:.2f} €", delta="Expected Amount")

    # Risk visualization
    risk_level = 100 - prediction_prob if prediction == 0 else prediction_prob
    st.write("Risk Level")
    st.progress(int(risk_level))

    st.write("")
    
    # Only create SHAP dashboard if not already in session state
    if "shap_result" not in st.session_state:
        customer_data = pd.DataFrame.from_dict({k: [v] for k, v in input_features.items()})
        result = create_clean_shap_dashboard(customer_data=customer_data)
        st.session_state.shap_result = result
        st.session_state.customer_data = customer_data
        st.session_state.shap_values = result["shap_values"]
    
    result = st.session_state.shap_result

    st.subheader("Prediction Result")
    prediction = result["prediction"]
    probability = result["churn_probability"] * 100

    # Display prediction with formatting
    if prediction == "Churn":
        st.error(f"Customer is predicted to churn with {probability:.1f}% probability")
    else:
        st.success(f"Customer is predicted to stay with {(100-probability):.1f}% probability")
    
    # Display the plot
    st.subheader("Feature Impact Analysis")
    st.pyplot(result["plot"])
    st.write("")
    st.write("")
    st.info("👉Now, you can go to '📖 Explain' page or '💡 Recommendations' or 📑 Generate Report page for further actions for this customer in the Navigation bar.")
    st.write("")