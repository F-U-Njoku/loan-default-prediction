import pickle
import pandas as pd
import numpy as np
import xgboost as xgb


def load_test_data(customer_id):
    df_demo = pd.read_csv("data/testdemographics.csv").loc[:,["customerid",'longitude_gps', 'latitude_gps', 'birthdate', 'bank_account_type', 'bank_name_clients','employment_status_clients']]
    df_prev = pd.read_csv("data/testprevloans.csv").loc[:,["customerid",'loannumber', 'totaldue', 'termdays', 'firstrepaiddate', 
'firstduedate']]
    df_perf = pd.read_csv("data/testperf.csv").loc[:,["customerid",'totaldue', 'termdays']]
    
    
    df_demo = df_demo[df_demo['customerid'] == customer_id]
    df_prev = df_prev[df_prev['customerid'] == customer_id]
    df_perf = df_perf[df_perf['customerid'] == customer_id]
    print(df_demo.head())
    print(df_prev.head())
    print(df_perf.head())
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
    print(grouped_prev.head())
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
    
    drop_cols = [ 'birthdate', "customerid" ]
    df = df.drop(columns=drop_cols)
    return df.dropna()

def main():
    customer_id = input("Enter customer ID: ")
    
    with open("model_5_4_100_01.bin", "rb") as f:
        model = pickle.load(f)
    
 
    df_demo, df_prev, df_perf = load_test_data(customer_id)
    if df_demo.empty or df_prev.empty or df_perf.empty:
        print(f"No data found for customer ID: {customer_id}")
        return

    df_demo = preprocess_demo(df_demo)
    df_prev = preprocess_prev(df_prev)
    df_test = combine_data(df_perf, df_prev, df_demo)
    df_test = feature_engineering(df_test)

    X_test = xgb.DMatrix(df_test)
    prediction_prob = model.predict(X_test)[0]
    prediction = "Good" if prediction_prob > 0.5 else "Bad"

    print(f"Customer {customer_id}:")
    print(f"Probability of Good: {prediction_prob:.2f}")
    print(f"Prediction: {prediction}")

    

if __name__ == "__main__":
    main()