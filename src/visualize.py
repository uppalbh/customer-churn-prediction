import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

sns.set_style("whitegrid")

def plot_feature_distributions(df, numerical_cols):
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{col}_distribution.png", dpi=300)
        plt.close()

  def plot_churn_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["Churn"])
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    
    plt.savefig("outputs/plots/churn_distribution.png", dpi=300)
    plt.close()


def plot_smote_impact(y_train, y_train_smote):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.countplot(x=y_train, ax=axes[0])
    axes[0].set_title("Before SMOTE")
    
    sns.countplot(x=y_train_smote, ax=axes[1])
    axes[1].set_title("After SMOTE")
    
    plt.tight_layout()
    plt.savefig("outputs/plots/smote_impact.png", dpi=300)
    plt.close()


def plot_cv_results(cv_results):
    model_names = list(cv_results.keys())
    means = [np.mean(scores) for scores in cv_results.values()]
    stds = [np.std(scores) for scores in cv_results.values()]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, means, yerr=stds, capsize=5)
    
    for bar, score in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.005,
            f"{score:.3f}",
            ha="center"
        )
        
    plt.ylabel("Accuracy")
    plt.title("Cross Validation Comparison")
    plt.tight_layout()
    
    plt.savefig("outputs/plots/model_comparison.png", dpi=300)
    plt.close()


def plot_roc_curves(models, X_train_smote, y_train_smote, X_test, y_test):
    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        model.fit(X_train_smote, y_train_smote)
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curves")
    plt.tight_layout()
    
    plt.savefig("outputs/plots/roc_curves.png", dpi=300)
    plt.close()


def plot_precision_recall(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    
    plt.savefig("outputs/plots/precision_recall.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    plt.savefig("outputs/plots/confusion_matrix.png", dpi=300)
    plt.close()


def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    
    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })
    feature_df = feature_df.sort_values("Importance", ascending=False).head(10)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_df, x="Importance", y="Feature")
    plt.title("Top XGBoost Features")
    plt.tight_layout()
    
    plt.savefig("outputs/plots/feature_importance.png", dpi=300)
    plt.close()
    return feature_df


def plot_churn_drivers(feature_df):
    top = feature_df.head(6)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top, x="Importance", y="Feature")
    plt.title("Top Business Churn Drivers")
    plt.tight_layout()
    
    plt.savefig("outputs/plots/churn_drivers.png", dpi=300)
    plt.close()


def plot_revenue_at_risk(results_df):
    top = results_df.nlargest(15, "Revenue_At_Risk")
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top)), top["Revenue_At_Risk"])
    plt.yticks(range(len(top)), [f"${x:.0f}" for x in top["MonthlyCharges"]])
    plt.title("Revenue At Risk Dashboard")
    plt.tight_layout()
    
    plt.savefig("outputs/plots/revenue_at_risk.png", dpi=300)
    plt.close()


def export_high_risk_customers(results_df):
    high_risk = results_df.nlargest(20, "Revenue_At_Risk")
    high_risk.to_csv("outputs/reports/high_risk_customers.csv", index=False)


def revenue_exposure_report(results_df):
    report = f"""
Revenue Exposure Analysis
=========================

Total Revenue At Risk:
${results_df['Revenue_At_Risk'].sum():,.2f}

Average Revenue At Risk:
${results_df['Revenue_At_Risk'].mean():,.2f}

Top 5 Customers Risk:
${results_df.nlargest(5, 'Revenue_At_Risk')['Revenue_At_Risk'].sum():,.2f}

Top 15 Customers Risk:
${results_df.nlargest(15, 'Revenue_At_Risk')['Revenue_At_Risk'].sum():,.2f}
"""

    with open("outputs/reports/revenue_exposure_report.txt", "w") as f:
        f.write(report)
