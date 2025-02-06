import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px

st.set_page_config(layout="wide")

# Custom CSS for both light and dark modes
st.markdown("""
<style>
    .metric-container {
        background-color: var(--background-color);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--primary-color);
    }
    .row-widget.stButton > button {
        width: 100%;
        background-color: var(--primary-color);
    }
    div[data-testid="stMetric"] {
        text-align: center;
        align-items: center;
        justify-content: center;
        display: flex;
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)


def load_test_data(customer_id):
    df_demo = pd.read_csv("data/testdemographics.csv").loc[:,["customerid",'longitude_gps', 'latitude_gps', 'birthdate', 'bank_account_type', 'bank_name_clients','employment_status_clients']]
    df_prev = pd.read_csv("data/testprevloans.csv").loc[:,["customerid",'loannumber', 'totaldue', 'termdays', 'firstrepaiddate', 'firstduedate']]
    df_perf = pd.read_csv("data/testperf.csv").loc[:,["customerid",'totaldue', 'termdays']]
    
    df_demo = df_demo[df_demo['customerid'] == customer_id]
    df_prev = df_prev[df_prev['customerid'] == customer_id]
    df_perf = df_perf[df_perf['customerid'] == customer_id]
    return df_demo, df_prev, df_perf

def preprocess_demo(df_demo):
    df_demo['birthdate'] = pd.to_datetime(df_demo['birthdate']).dt.year
    df_demo["employment_status_clients"] = df_demo["employment_status_clients"].fillna("UNKNOWN")
    return df_demo

def preprocess_prev(df_prev):
    date_cols = ['firstduedate', 'firstrepaiddate']
    for col in date_cols:
        df_prev[col] = pd.to_datetime(df_prev[col])
    
    df_prev["first_payment_default"] = ((df_prev.firstrepaiddate.dt.normalize() - 
                                       df_prev.firstduedate.dt.normalize()).dt.days > 0).astype(int)
    
    grouped_prev = df_prev.groupby('customerid').agg({
        'loannumber': ['max'],
        'totaldue': 'min',
        'termdays': 'min',
        'first_payment_default': 'sum'
    }).reset_index()
    
    grouped_prev.columns = ['_'.join(col).strip('_') for col in grouped_prev.columns]
    grouped_prev.rename(columns={'customerid_': 'customerid'}, inplace=True)
    return grouped_prev

def combine_data(df_perf, grouped_prev, df_demo):
    df_loans = pd.merge(df_perf, grouped_prev, on="customerid", how='left')
    df_test = pd.merge(df_loans, df_demo, on="customerid", how='left')
    return df_test


def feature_engineering(df):
    df["age"] = 2024 - df['birthdate']
    banks_of_interest = ['GT Bank', 'First Bank', 'Access_Diamond', 'UBA', 'Zenith Bank']
    status_of_interest = ['Permanent', 'Self-Employed']
    
    df['bank_name_clients'] = df['bank_name_clients'].replace(
        {'Diamond Bank': 'Access_Diamond', 'Access Bank': 'Access_Diamond'}
    )
    bank_encoded = pd.get_dummies(df['bank_name_clients'], dtype=float)
    for bank in banks_of_interest:
        if bank not in bank_encoded.columns:
            bank_encoded[bank] = 0
    bank_encoded = bank_encoded[banks_of_interest].add_prefix('is_')
    
    status_encoded = pd.get_dummies(df['employment_status_clients'], dtype=float)
    for status in status_of_interest:
        if status not in status_encoded.columns:
            status_encoded[status] = 0
    status_encoded = status_encoded[status_of_interest].add_prefix('is_')
    
    df = pd.concat([df, bank_encoded, status_encoded], axis=1)
    df['bank_account_num'] = np.where(df['bank_account_type'] == 'Savings', 1, 0)
    df = df.drop(columns=['bank_account_type', 'bank_name_clients', 'employment_status_clients'])
    
    drop_cols = ['birthdate', "customerid"]
    df = df.drop(columns=drop_cols)
    return df.dropna()


# Main app
st.title("🏦 Loan Default Risk Assessment")
st.markdown("---")

with st.sidebar:
    st.header("About")
    st.info("""
    This application predicts loan default risk based on:
    - Customer Demographics
    - Previous Loan History
    - Current Performance
    """)

customer_id = st.text_input("🆔 Enter Customer ID:", placeholder="e.g., 8a858899538ddb8e015390510b321f08")

if st.button("🔍 Analyze Customer"):
    if customer_id:
        try:
            with st.spinner("Analyzing customer data..."):
                with open("model_5_4_100_01.bin", "rb") as f:
                    model = pickle.load(f)

                df_demo, df_prev, df_perf = load_test_data(customer_id)
                
                if df_demo.empty or df_prev.empty or df_perf.empty:
                    st.error("❌ No data found for this customer ID")
                else:
                    df_demo = preprocess_demo(df_demo)
                    df_prev = preprocess_prev(df_prev)
                    df_test = combine_data(df_perf, df_prev, df_demo)
                    
                    # Display customer profile
                    st.subheader("📊 Customer Profile")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Age", f"{2024 - df_demo['birthdate'].iloc[0]} years")
                        st.metric("Bank", df_demo['bank_name_clients'].iloc[0])
                    
                    with col2:
                        st.metric("Account Type", df_demo['bank_account_type'].iloc[0])
                        st.metric("Employment", df_demo['employment_status_clients'].iloc[0])
                    
                    with col3:
                        st.metric("Number of Loans", len(df_prev))
                        st.metric("Payment Defaults", df_prev['first_payment_default_sum'])

                    

                    # Model prediction
                    df_model = feature_engineering(df_test)
                    X_test = xgb.DMatrix(df_model)
                    prediction_prob = model.predict(X_test)[0]
                    prediction = "Good" if prediction_prob > 0.5 else "Bad"

                    
                    
                    risk_col1, risk_col2 = st.columns(2)
                    with risk_col1:
                        # Location
                        if not df_demo['longitude_gps'].isna().all():
                            st.subheader("📍 Customer Location")
                            map_data = pd.DataFrame({
                                'lat': [df_demo['latitude_gps'].iloc[0]],
                                'lon': [df_demo['longitude_gps'].iloc[0]]
                            })
                            st.map(map_data)
                        
                        
                    
                    with risk_col2:
                        st.subheader("🎯 Risk Assessment")
                        # Determine risk level and corresponding styling
                        if prediction_prob > 0.7:
                            risk_level = "Low Risk"
                            symbol = "↓"
                            color = "green"
                        elif prediction_prob < 0.5:
                            risk_level = "High Risk"
                            symbol = "↑"
                            color = "red"
                        else:
                            risk_level = "Medium Risk"
                            symbol = "~"
                            color = "yellow"
                        
                        # Display the risk metric
                        st.metric(
                            label="Probability to not default",
                            value=f"{prediction_prob:.1%}",
                        )
                        
                        # Custom colored risk level
                        st.markdown(
                            f"""
                            <div style="text-align: center; color: {color}; font-size: 18px; font-weight: bold;">
                                {symbol} {risk_level}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        gauge_chart = px.pie(values=[prediction_prob, 1-prediction_prob], 
                                          names=['Good', 'Bad'],
                                          hole=0.7,
                                          color_discrete_sequence=['#4CAF50', '#f44336'])
                        gauge_chart.update_layout(showlegend=False)
                        st.plotly_chart(gauge_chart)
                        
                        

                    if prediction == "Good":
                        st.success("✅ This customer is predicted to be a low-risk loan candidate.")
                    else:
                        st.warning("⚠️ This customer is predicted to be a high-risk loan candidate.")
                        
        except Exception as e:
            st.error(f"Error analyzing customer data: {str(e)}")
    else:
        st.warning("⚠️ Please enter a Customer ID")