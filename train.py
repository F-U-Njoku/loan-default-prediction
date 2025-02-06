import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
def load_data():
    df_demo = pd.read_csv("data/traindemographics.csv").drop_duplicates()
    df_prev = pd.read_csv("data/trainprevloans.csv").drop_duplicates()
    df_perf = pd.read_csv("data/trainperf.csv").drop_duplicates()
    return df_demo, df_prev, df_perf

# Preprocess demographic data
def preprocess_demo(df_demo):
    df_demo['birthdate'] = pd.to_datetime(df_demo['birthdate']).dt.year
    df_demo = df_demo.drop(columns=['bank_branch_clients', 'level_of_education_clients'])
    df_demo["employment_status_clients"] = df_demo["employment_status_clients"].fillna("UNKNOWN")
    return df_demo

# Preprocess previous loans data
def preprocess_prev(df_prev):
    df_prev = df_prev.drop(columns=['referredby'])
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        df_prev[col] = pd.to_datetime(df_prev[col])
    
    # Feature engineering
    df_prev["first_payment_default"] = ((df_prev.firstrepaiddate.dt.normalize() - 
                                         df_prev.firstduedate.dt.normalize()).dt.days > 0).astype(int)
    df_prev["loan_default"] = ((df_prev.closeddate.dt.normalize() - 
                                df_prev.approveddate.dt.normalize()).dt.days > df_prev.termdays).astype(int)
    
    # Aggregation
    grouped_prev = df_prev.groupby('customerid').agg({
        'loannumber': 'max',
        'loanamount': ['min', 'mean', 'max'],
        'totaldue': ['min', 'mean', 'max'],
        'termdays': ['min', 'mean', 'max'],
        'loan_default': 'sum',
        'first_payment_default': 'sum'
    }).reset_index()
    
    # Flatten column names
    grouped_prev.columns = ['_'.join(col).strip('_') for col in grouped_prev.columns]
    grouped_prev.rename(columns={'customerid_': 'customerid'}, inplace=True)
    return grouped_prev

# Preprocess performance data
def preprocess_perf(df_perf):
    df_perf = df_perf.drop(columns=['referredby'])
    df_perf['approveddate'] = pd.to_datetime(df_perf['approveddate']).dt.date
    df_perf['creationdate'] = pd.to_datetime(df_perf['creationdate']).dt.date
    df_perf["good_bad_flag"] = df_perf.good_bad_flag.map({"Good": 1, "Bad": 0})
    return df_perf

# Combine data
def combine_data(df_perf, grouped_prev, df_demo):
    df_loans = pd.merge(df_perf, grouped_prev, on="customerid", how='left')
    df_train = pd.merge(df_loans, df_demo, on="customerid", how='left')
    return df_train

# Feature engineering and encoding
def feature_engineering(df):
    df["age"] = 2024 - df['birthdate']
    banks_of_interest = ['GT Bank', 'First Bank', 'Access_Diamond', 'UBA', 'Zenith Bank']
    status_of_interest = ['Permanent', 'Self-Employed']
    
    # Replace and encode bank names
    df['bank_name_clients'] = df['bank_name_clients'].replace(
        {'Diamond Bank': 'Access_Diamond', 'Access Bank': 'Access_Diamond'}
    )
    bank_encoded = pd.get_dummies(df['bank_name_clients'], dtype=float)
    bank_encoded = bank_encoded[banks_of_interest].add_prefix('is_')
    
    # Encode employment status
    status_encoded = pd.get_dummies(df['employment_status_clients'], dtype=float)
    status_encoded = status_encoded[status_of_interest].add_prefix('is_')
    
    # Combine encodings
    df = pd.concat([df, bank_encoded, status_encoded], axis=1)
    df['bank_account_num'] = np.where(df['bank_account_type'] == 'Savings', 1, 0)
    df = df.drop(columns=['bank_account_type', 'bank_name_clients', 'employment_status_clients'])
    
    # Drop redundant columns
    drop_cols = ['approveddate', 'creationdate', 'birthdate', 'loanamount', 'loanamount_max', 
                 'loanamount_mean', 'loanamount_min', 'totaldue_mean', 'totaldue_max', 
                 'loan_default_sum', 'termdays_max', 'termdays_mean', 'loannumber', "customerid", "systemloanid"]
    df = df.drop(columns=drop_cols)
    return df.dropna()

# Train the model
def train(X, y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set up XGBoost parameters
    params = {
        'max_depth': 3,
        'min_child_weight': 4,
        'gamma': 0.1,
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss'
    }
    
    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=14)
    
    # Predict on test set
    y_pred_prob = model.predict(dtest)  # Probabilities
    y_pred_binary = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions
    
    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    
    return model

def main():
    # Load and preprocess data
    df_demo, df_prev, df_perf = load_data()
    df_demo = preprocess_demo(df_demo)
    df_prev = preprocess_prev(df_prev)
    df_perf = preprocess_perf(df_perf)
    df_train = combine_data(df_perf, df_prev, df_demo)
    df_train = feature_engineering(df_train)
    
    # Split data
    X = df_train.drop(columns=["good_bad_flag"])
    y = df_train["good_bad_flag"]

    print(X.dtypes)
    
    # Train and save model
    output_file = "model_5_4_100_01.bin"
    model = train(X, y)
    with open(output_file, "wb") as f:
        pickle.dump(model, f)
    print("Model saved to model_5_4_100_01.bin")

if __name__ == "__main__":
    main()
