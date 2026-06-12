import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier


def compare_models(X_train_smote, y_train_smote):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score( model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
        cv_results[name] = scores
    return models, cv_results


def train_xgboost(X_train_smote, y_train_smote):
    model = XGBClassifier( random_state=42, eval_metric="logloss")
    model.fit(X_train_smote, y_train_smote)
    with open("models/XGB_final_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def train_logistic_regression(X_train_log, y_train_smote):
    model = LogisticRegression(random_state=42)
    model.fit(X_train_log, y_train_smote)
    return model
