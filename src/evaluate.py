import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_val, y_val, X_test, y_test):
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    return y_test_pred


def revenue_at_risk(log_model, X_test_log, X_test):
    churn_prob = log_model.predict_proba(X_test_log)[:, 1]
    results_df = pd.DataFrame({
        "MonthlyCharges": X_test["MonthlyCharges"].values,
        "Churn_Probability": churn_prob
    })
    results_df["Revenue_At_Risk"] = ( results_df["MonthlyCharges"] * results_df["Churn_Probability"])
    return results_df
