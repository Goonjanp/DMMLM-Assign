import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_loan_approval(input_data):
    # Assuming your input data is a dictionary with the same keys as your features
    df = pd.DataFrame([input_data])

    # Encode categorical features using LabelEncoder (you might need to adjust this based on your dataset)
    le = LabelEncoder()
    for column in ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']:
        if column in df.columns:
            df[column] = le.fit_transform(df[column])


    # Make predictions using the loaded model
    prediction = model.predict(df)
    return prediction[0]

# Streamlit app
st.title("Loan Approval Prediction")

st.write("Enter the details to predict loan approval.")

# Input fields for user data
employment_status = st.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Unemployed'])
education_level = st.selectbox("Education Level", ['Graduate', 'High School', 'Undergraduate'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
home_ownership_status = st.selectbox("Home Ownership Status", ['Owned', 'Rented', 'Mortgaged'])
loan_purpose = st.selectbox("Loan Purpose", ['Debt Consolidation', 'Home Improvement', 'Business'])
income = st.number_input("Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.number_input("Credit History (0-100)", min_value=0, max_value=100)


if st.button("Predict"):
    input_data = {
        'EmploymentStatus': employment_status,
        'EducationLevel': education_level,
        'MaritalStatus': marital_status,
        'HomeOwnershipStatus': home_ownership_status,
        'LoanPurpose': loan_purpose,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditHistory': credit_history
    }

    prediction = predict_loan_approval(input_data)

    if prediction == 1:
        st.success("Loan is likely to be approved.")
    else:
        st.error("Loan is likely to be denied.")