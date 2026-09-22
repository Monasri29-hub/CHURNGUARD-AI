import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CHURNGUARD AI", page_icon="📉")
model = joblib.load("churnguard_model.joblib")

st.title("CHURNGUARD AI")
st.caption("Customer Churn Risk Predictor")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=840.0)

def risk_level(p):
    if p < 0.30:
        return "LOW"
    if p < 0.60:
        return "MEDIUM"
    return "HIGH"

if st.button("Predict Churn Risk"):
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probability = float(model.predict_proba(customer)[0, 1])
    risk = risk_level(probability)

    st.metric("Churn Probability", f"{probability * 100:.1f}%")
    st.write("**Prediction:**", "Likely to churn" if probability >= 0.50 else "Likely to stay")
    st.write("**Risk Level:**", risk)
