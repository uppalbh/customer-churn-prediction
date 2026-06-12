import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Machine Learning & Processing Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE

# =====================================================================
# 1. SETUP & UTILITY FUNCTIONS
# =====================================================================
sns.set_style("whitegrid")

# Create output directories
PLOTS_DIR = Path("outputs/plots")
REPORTS_DIR = Path("outputs/reports")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(filename):
    """Formats, saves, and closes the active matplotlib plot."""
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / Path(filename).name, dpi=300, bbox_inches="tight")
    plt.close()

def load_data_safely(data_path):
    """Load data from various formats (CSV, pickle, parquet, etc.)"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    # First, check file signature to determine format
    with open(data_path, 'rb') as f:
        header = f.read(20)
        
        # Check for pickle format (starts with pickle protocol markers)
        if header.startswith(b'\x80\x04') or header.startswith(b'\x80\x03') or b'pickle' in header:
            print("Detected pickle format, loading with pickle...")
            try:
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, pd.DataFrame):
                    return data
                else:
                    return pd.DataFrame(data)
            except Exception as e:
                print(f"Error loading pickle: {e}")
        
        # Check for parquet format
        if header.startswith(b'PAR1'):
            print("Detected parquet format...")
            try:
                return pd.read_parquet(data_path)
            except Exception as e:
                print(f"Error loading parquet: {e}")
        
        # Check for feather format
        if header.startswith(b'ARROW1'):
            print("Detected feather format...")
            try:
                return pd.read_feather(data_path)
            except Exception as e:
                print(f"Error loading feather: {e}")
    
    # Try CSV with different delimiters and encodings
    csv_configs = [
        {'delimiter': ',', 'encoding': 'utf-8'},
        {'delimiter': ',', 'encoding': 'latin1'},
        {'delimiter': ',', 'encoding': 'iso-8859-1'},
        {'delimiter': ';', 'encoding': 'utf-8'},
        {'delimiter': '\t', 'encoding': 'utf-8'},
        {'delimiter': '|', 'encoding': 'utf-8'},
        {'delimiter': ',', 'encoding': 'cp1252'},
        {'delimiter': ',', 'encoding': 'utf-16'},
    ]
    
    for config in csv_configs:
        try:
            print(f"Trying CSV with delimiter='{config['delimiter']}', encoding={config['encoding']}")
            df = pd.read_csv(
                data_path, 
                delimiter=config['delimiter'],
                encoding=config['encoding'],
                on_bad_lines='skip',  # Skip problematic lines
                engine='python'  # More forgiving parser
            )
            if len(df.columns) > 1:  # Valid CSV should have multiple columns
                print(f"Successfully loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")
                return df
        except Exception as e:
            continue
    
    # If all else fails, try reading as text and manual parsing
    print("Attempting manual text parsing...")
    try:
        with open(data_path, 'r', encoding='latin1') as f:
            lines = f.readlines()
        
        # Find lines with reasonable content
        valid_lines = []
        for line in lines[:100]:  # Check first 100 lines
            if ',' in line and len(line.strip()) > 10:
                valid_lines.append(line.strip())
        
        if valid_lines:
            # Write clean lines to temporary file
            temp_path = Path("temp_clean_data.csv")
            with open(temp_path, 'w') as f:
                f.write('\n'.join(valid_lines))
            df = pd.read_csv(temp_path)
            os.remove(temp_path)
            return df
    except Exception as e:
        print(f"Manual parsing failed: {e}")
    
    raise Exception(f"Could not load data from {data_path}. File may be corrupted or in unsupported format.")

def inspect_file_content(file_path):
    """Inspect file content to help debug"""
    print(f"\n=== File Inspection: {file_path} ===")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    with open(file_path, 'rb') as f:
        raw_data = f.read(500)
        print(f"First 100 bytes (hex): {raw_data[:100].hex()}")
        print(f"First 100 bytes (ascii): {raw_data[:100]}")
        
        # Try to decode as text
        try:
            text_data = raw_data.decode('utf-8', errors='ignore')
            print(f"First 200 chars as text: {text_data[:200]}")
        except:
            print("Cannot decode as UTF-8 text")
    
    print("="*50)

# =====================================================================
# 2. DEFINING PLOTTING & REPORTING FUNCTIONS
# =====================================================================
def plot_feature_distributions(df, numerical_cols):
    for col in numerical_cols:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            save_plot(f"{col}_distribution.png")

def plot_churn_distribution(df):
    if "Churn" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df["Churn"])
        plt.title("Customer Churn Distribution")
        save_plot("churn_distribution.png")

def plot_smote_impact(y_train, y_train_smote):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=y_train, ax=axes[0])
    axes[0].set_title("Before SMOTE")
    sns.countplot(x=y_train_smote, ax=axes[1])
    axes[1].set_title("After SMOTE")
    save_plot("smote_impact.png")

def plot_cv_results(cv_results):
    if not cv_results:
        return
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
    save_plot("model_comparison.png")

def plot_roc_curves(models, X_train_smote, y_train_smote, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        # Clean training step without output leakage
        model.fit(X_train_smote, y_train_smote)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} ({roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curves")
    save_plot("roc_curves.png")

def plot_precision_recall(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.title("Precision Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        save_plot("precision_recall.png")

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    save_plot("confusion_matrix.png")

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(10)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(data=feature_df, x="Importance", y="Feature")
        plt.title("Top XGBoost Features")
        save_plot("feature_importance.png")
        return feature_df
    return pd.DataFrame()

def plot_churn_drivers(feature_df):
    if not feature_df.empty:
        top = feature_df.head(6)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=top, x="Importance", y="Feature")
        plt.title("Top Business Churn Drivers")
        save_plot("churn_drivers.png")

def plot_revenue_at_risk(results_df):
    if not results_df.empty:
        top = results_df.nlargest(15, "Revenue_At_Risk")
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top)), top["Revenue_At_Risk"])
        plt.yticks(range(len(top)), [f"${x:.0f}" for x in top["MonthlyCharges"]])
        plt.title("Revenue At Risk Dashboard")
        save_plot("revenue_at_risk.png")

def export_high_risk_customers(results_df):
    if not results_df.empty:
        high_risk = results_df.nlargest(20, "Revenue_At_Risk")
        high_risk.to_csv(REPORTS_DIR / "high_risk_customers.csv", index=False)

def revenue_exposure_report(results_df):
    if results_df.empty:
        return
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
    with open(REPORTS_DIR / "revenue_exposure_report.txt", "w") as f:
        f.write(report)

# =====================================================================
# 3. PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    
    # --- Step 1: Load Dataset ---
    data_path = "../../data/processed_data.csv"
    
    # Try multiple possible paths
    possible_paths = [
        data_path,
        Path(__file__).parent / "../../data/processed_data.csv",
        Path(__file__).parent / "../../data/data.csv",
        Path(__file__).parent / "../data/processed_data.csv",
        Path(__file__).parent / "data/processed_data.csv",
        "customer_churn_data.csv",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",  # Common Telco dataset filename
    ]
    
    data_path = None
    for path in possible_paths:
        path = Path(path).resolve()
        if path.exists():
            data_path = str(path)
            print(f"Found data file: {data_path}")
            break
    
    if data_path is None:
        # Search for CSV files in current directory and subdirectories
        print("\nSearching for CSV files...")
        csv_files = list(Path('.').rglob('*.csv'))
        if csv_files:
            print(f"Found CSV files:")
            for f in csv_files[:10]:  # Show first 10
                print(f"  - {f}")
            data_path = str(csv_files[0])
            print(f"\nUsing: {data_path}")
        else:
            raise FileNotFoundError("No data file found. Please ensure your dataset is in the correct location.")
    
    # Inspect file before loading
    inspect_file_content(data_path)
    
    print(f"\nLoading data from: {data_path}")
    df = load_data_safely(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop customerID if exists
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    
    # Save a clean baseline for original un-encoded columns before mapping transforms
    if "MonthlyCharges" in df.columns:
        original_monthly_charges = df["MonthlyCharges"].copy()
    else:
        original_monthly_charges = pd.Series(dtype=float)
        print("Warning: 'MonthlyCharges' column not found")

    # --- Step 2: Data Cleaning ---
    print("\nCleaning data...")
    nonCatCol = ["tenure", "MonthlyCharges", "TotalCharges"]
    # Filter to only existing columns
    nonCatCol = [col for col in nonCatCol if col in df.columns]
    
    # Replace empty strings with 0
    df = df.replace(r'^\s*$', 0.0, regex=True)
    
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0.0)
    
    # Fire data distribution plots before mathematical alterations
    plot_feature_distributions(df, nonCatCol)
    if "Churn" in df.columns:
        plot_churn_distribution(df)
    
    # Transform skewness
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = np.sqrt(df["TotalCharges"] + 1e-9)  # Add small epsilon to avoid sqrt(0)

    # --- Step 3: Categorical Encoding ---
    print("\nEncoding categorical attributes...")
    encoders = {}
    for col in df.columns:
        if col not in nonCatCol and col != "Churn" and df[col].dtype == 'object':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder

    with open("encoders_data.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # --- Step 4: Stratified Splitting & Class Balancing via SMOTE ---
    print("\nSplitting datasets and applying SMOTE...")
    if "Churn" not in df.columns:
        raise KeyError("'Churn' column not found in dataframe")
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Document balance updates
    plot_smote_impact(y_train, y_train_smote)

    # --- Step 5: Feature Scaling for Linear Models ---
    X_train_log = X_train_smote.copy()
    X_val_log = X_val.copy()
    X_test_log = X_test.copy()

    scalers = {}
    for col in nonCatCol:
        if col in X_train_log.columns:
            scaler = StandardScaler()
            X_train_log[col] = scaler.fit_transform(X_train_log[[col]])
            X_val_log[col] = scaler.transform(X_val_log[[col]])
            X_test_log[col] = scaler.transform(X_test_log[[col]])
            scalers[col] = scaler

    # --- Step 6: Cross-Validation Testing ---
    print("\nCalculating Cross-Validation scores...")
    base_models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    }

    cv_results = {}
    for name, model in base_models.items():
        scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
        cv_results[name] = scores
        print(f"{name}: Mean accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    plot_cv_results(cv_results)

    # --- Step 7: Train Final Classifier Metrics ---
    print("\nTraining final classifiers for analysis plots...")
    xgb_final = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    xgb_final.fit(X_train_smote, y_train_smote)
    y_pred_xgb = xgb_final.predict(X_test)
    
    print(f"\nXGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_xgb))
    
    # Evaluation charts
    plot_roc_curves(base_models, X_train_smote, y_train_smote, X_test, y_test)
    plot_precision_recall(xgb_final, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred_xgb)
    
    # Feature configurations
    feat_df = plot_feature_importance(xgb_final, X_train_smote.columns)
    plot_churn_drivers(feat_df)

    # --- Step 8: Logistic Regression & Revenue Exposure Analysis ---
    print("\nAnalyzing Financial Risks via Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_log, y_train_smote)
    
    # Calculate extraction risks using validation arrays
    val_probabilities = log_reg.predict_proba(X_val_log)[:, 1]
    
    # Map back variables to compile clear values
    if len(original_monthly_charges) > 0 and len(X_val.index) > 0:
        financial_df = pd.DataFrame({
            "MonthlyCharges": original_monthly_charges.iloc[X_val.index].values,
            "Churn_Probability": val_probabilities
        })
        financial_df["Revenue_At_Risk"] = financial_df["MonthlyCharges"] * financial_df["Churn_Probability"]
        
        # Save Financial reporting distributions
        plot_revenue_at_risk(financial_df)
        export_high_risk_customers(financial_df)
        revenue_exposure_report(financial_df)
    else:
        print("Warning: Could not perform revenue analysis due to missing data")

    print("\n" + "="*50)
    print("SUCCESS: Execution completed cleanly!")
    print(f"-> Graphical plots exported safely to: '{PLOTS_DIR}/'")
    print(f"-> Text & CSV business files exported to: '{REPORTS_DIR}/'")
    print("="*50)