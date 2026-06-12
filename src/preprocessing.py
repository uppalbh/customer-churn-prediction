import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=["customerID"], inplace=True)
    return df


def clean_data(df):
    df = df.replace({" ": 0.0})

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["TotalCharges"] = np.sqrt(df["TotalCharges"])

    return df


def encode_features(df):
    encoders = {}

    for col in df.columns:
        if col not in NUM_COLS:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder

    with open("models/encoders_data.pkl", "wb") as f:
        pickle.dump(encoders, f)

    return df


def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)


def scale_for_logistic(X_train_smote, X_val, X_test):
    X_train_log = X_train_smote.copy()
    X_val_log = X_val.copy()
    X_test_log = X_test.copy()

    scalers = {}

    for col in NUM_COLS:
        scaler = StandardScaler()

        X_train_log[col] = scaler.fit_transform(X_train_log[[col]])
        X_val_log[col] = scaler.transform(X_val_log[[col]])
        X_test_log[col] = scaler.transform(X_test_log[[col]])

        scalers[col] = scaler

    return X_train_log, X_val_log, X_test_log, scalers
