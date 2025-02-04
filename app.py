import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

def load_model():
    # Load your trained model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_input(data):
    # Convert input data to match model's expected format
    # Add any necessary preprocessing steps here
    return data

def predict_loan_status(model, features):
    # Make prediction using the model
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction[0], probability[0]

def main():
    st.title('Loan Default Prediction System')
    st.write('Enter the customer information to predict loan default probability')
    
    # Create input form
    with st.form("loan_prediction_form"):
        # Customer Information
        st.subheader("Customer Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=18, max_value=100)
            loan_amount = st.number_input('Loan Amount', min_value=0)
            term_days = st.number_input('Term Days', min_value=0)
            
        with col2:
            total_due = st.number_input('Total Due', min_value=0)
            latitude = st.number_input('Latitude GPS', min_value=-90.0, max_value=90.0)
            longitude = st.number_input('Longitude GPS', min_value=-180.0, max_value=180.0)
        
        # Additional Features
        st.subheader("Loan History")
        col3, col4 = st.columns(2)
        
        with col3:
            loan_number = st.number_input('Loan Number', min_value=1)
            previous_loans = st.number_input('Number of Previous Loans', min_value=0)
            
        with col4:
            previous_defaults = st.number_input('Previous Defaults', min_value=0)
            first_payment_defaults = st.number_input('First Payment Defaults', min_value=0)
        
        # Submit button
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            try:
                # Load model
                model = load_model()
                
                # Prepare input data
                input_data = {
                    'age': age,
                    'loan_amount': loan_amount,
                    'term_days': term_days,
                    'total_due': total_due,
                    'latitude_gps': latitude,
                    'longitude_gps': longitude,
                    'loan_number': loan_number,
                    'previous_loans': previous_loans,
                    'previous_defaults': previous_defaults,
                    'first_payment_defaults': first_payment_defaults
                }
                
                # Convert to DataFrame and preprocess
                features = pd.DataFrame([input_data])
                features = preprocess_input(features)
                
                # Make prediction
                prediction, probability = predict_loan_status(model, features)
                
                # Display results
                st.subheader('Prediction Results')
                
                # Create columns for results
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric(
                        label="Prediction",
                        value="Default Risk" if prediction == 1 else "Good Standing"
                    )
                
                with col_result2:
                    st.metric(
                        label="Default Probability",
                        value=f"{probability[1]:.2%}"
                    )
                
                # Add risk assessment visualization
                risk_color = 'red' if probability[1] > 0.5 else 'green'
                st.progress(float(probability[1]))
                
                # Additional insights
                st.subheader('Risk Assessment Details')
                risk_level = ''
                if probability[1] < 0.3:
                    risk_level = 'Low Risk'
                elif probability[1] < 0.7:
                    risk_level = 'Medium Risk'
                else:
                    risk_level = 'High Risk'
                    
                st.info(f'Risk Level: {risk_level}')
                
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
                
    # Add explanatory notes
    with st.expander("How to use this predictor"):
        st.write("""
        1. Fill in all the required information about the loan application
        2. Click 'Predict' to see the loan default risk assessment
        3. The system will provide:
           - A binary prediction (Default Risk/Good Standing)
           - The probability of default
           - A risk level assessment
        """)
        
    # Add footer
    st.markdown("---")
    st.markdown("Loan Default Prediction System © 2024")

if __name__ == '__main__':
    main()