# main.py
"""
Single-entry pipeline for Employee Attrition Prediction (for evaluators)
Run: python main.py
Outputs:
 - rf_model.joblib
 - svm_linear.joblib, svm_poly.joblib, svm_rbf.joblib
 - rf_confusion_matrix.png, svm_*_confusion_matrix.png, *_roc.png
 - preprocessed_train.csv, preprocessed_test.csv (if created)
 - Model_Comparison_Report.md
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

# ---------- Config ----------
INPUT_CSV = "Employee_Performance_Retention.csv"
TRAIN_OUT = "preprocessed_train.csv"
TEST_OUT = "preprocessed_test.csv"
SCALER_FILE = "scaler.joblib"
DEPT_COLUMNS_FILE = "dept_columns.joblib"

RF_MODEL_FILE = "rf_model.joblib"
SVM_LINEAR_FILE = "svm_linear.joblib"
SVM_POLY_FILE = "svm_poly.joblib"
SVM_RBF_FILE = "svm_rbf.joblib"

REPORT_MD = "Model_Comparison_Report.md"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# make plots prettier
sns.set(style="whitegrid")

# ---------- Helpers ----------
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in folder: {os.getcwd()}")
    return pd.read_csv(path)

def save_confusion(cm, outpath, title):
    plt.figure(figsize=(4.5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_roc(y_test, y_proba, outpath, title):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return roc_auc

def evaluate_model(model, X_test, y_test, name_prefix):
    # predictions
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:,1]
        except:
            y_proba = model.decision_function(X_test)
            # scale to 0-1
            y_proba = (y_proba - y_proba.min())/(y_proba.max()-y_proba.min()+1e-9)
    else:
        try:
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min())/(scores.max()-scores.min()+1e-9)
        except:
            y_proba = np.zeros_like(y_pred, dtype=float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test))>1 else 0.0

    # save plots
    cm = confusion_matrix(y_test, y_pred)
    save_confusion(cm, f"{name_prefix}_confusion_matrix.png", f"{name_prefix} - Confusion Matrix")
    save_roc(y_test, y_proba, f"{name_prefix}_roc.png", f"{name_prefix} - ROC Curve")

    return {"model_name": name_prefix, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}

# ---------- Preprocessing ----------
def preprocess(create_if_missing=True):
    # If preprocessed CSVs exist, load them
    if os.path.exists(TRAIN_OUT) and os.path.exists(TEST_OUT):
        print("Preprocessed CSVs found. Loading...")
        train = pd.read_csv(TRAIN_OUT)
        test = pd.read_csv(TEST_OUT)
        X_train = train.drop(columns=["Attrition"])
        y_train = train["Attrition"]
        X_test = test.drop(columns=["Attrition"])
        y_test = test["Attrition"]
        return X_train, X_test, y_train, y_test

    # Else, create preprocessing pipeline
    print("Running preprocessing... (this will create preprocessed_train.csv and preprocessed_test.csv)")
    df = load_csv(INPUT_CSV)
    # drop identifier if present
    if "Employee_ID" in df.columns:
        df = df.drop(columns=["Employee_ID"])

    # Target & binary encodings
    df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})
    df["Promotion_in_Last_2_Years"] = df["Promotion_in_Last_2_Years"].map({"Yes":1, "No":0})

    # job satisfaction ordinal
    sat_map = {"Low":0, "Medium":1, "High":2}
    df["Job_Satisfaction_Level"] = df["Job_Satisfaction_Level"].map(sat_map)
    if df["Job_Satisfaction_Level"].isnull().any():
        df["Job_Satisfaction_Level"] = df["Job_Satisfaction_Level"].fillna(df["Job_Satisfaction_Level"].median())

    # one-hot department
    dept_dummies = pd.get_dummies(df["Department"], prefix="Dept")
    df = pd.concat([df.drop(columns=["Department"]), dept_dummies], axis=1)

    # features / target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # scale numeric columns
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numeric_cols])
    X_test_num = scaler.transform(X_test[numeric_cols])

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = X_train_num
    X_test_scaled[numeric_cols] = X_test_num

    # save outputs
    train_out_df = X_train_scaled.copy(); train_out_df["Attrition"] = y_train.values
    test_out_df = X_test_scaled.copy(); test_out_df["Attrition"] = y_test.values
    train_out_df.to_csv(TRAIN_OUT, index=False)
    test_out_df.to_csv(TEST_OUT, index=False)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(list(dept_dummies.columns), DEPT_COLUMNS_FILE)
    print("Saved preprocessed files and scaler.")
    return X_train_scaled, X_test_scaled, y_train, y_test

# ---------- Train Random Forest ----------
def train_random_forest(X_train, y_train):
    print("Training Random Forest (baseline)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_MODEL_FILE)
    print("Random Forest saved:", RF_MODEL_FILE)
    return rf

# ---------- Train SVMs (light tuning) ----------
def train_svms(X_train, y_train):
    print("Training SVMs (linear, poly, rbf) - light tuning (quick)...")
    results = {}
    # linear
    svc_lin = SVC(kernel="linear", class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    param_lin = {"C":[0.1, 1]}
    gs_lin = GridSearchCV(svc_lin, param_lin, cv=3, n_jobs=-1, verbose=0)
    gs_lin.fit(X_train, y_train)
    best_lin = gs_lin.best_estimator_
    joblib.dump(best_lin, SVM_LINEAR_FILE)
    results["linear"] = best_lin
    print("svm_linear saved (best params):", gs_lin.best_params_)

    # poly (degree small)
    svc_poly = SVC(kernel="poly", class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    param_poly = {"C":[0.1, 1], "degree":[2,3]}
    gs_poly = GridSearchCV(svc_poly, param_poly, cv=3, n_jobs=-1, verbose=0)
    gs_poly.fit(X_train, y_train)
    best_poly = gs_poly.best_estimator_
    joblib.dump(best_poly, SVM_POLY_FILE)
    results["poly"] = best_poly
    print("svm_poly saved (best params):", gs_poly.best_params_)

    # rbf
    svc_rbf = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    param_rbf = {"C":[0.1, 1, 10], "gamma":["scale","auto"]}
    gs_rbf = GridSearchCV(svc_rbf, param_rbf, cv=3, n_jobs=-1, verbose=0)
    gs_rbf.fit(X_train, y_train)
    best_rbf = gs_rbf.best_estimator_
    joblib.dump(best_rbf, SVM_RBF_FILE)
    results["rbf"] = best_rbf
    print("svm_rbf saved (best params):", gs_rbf.best_params_)

    return results

# ---------- Build report ----------
def build_report(evals, rf_feature_importances=None):
    lines = []
    lines.append("# Model Comparison Report")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat()} UTC\n")
    lines.append("## Summary Table\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for e in evals:
        lines.append(f"| {e['model_name']} | {e['accuracy']:.4f} | {e['precision']:.4f} | {e['recall']:.4f} | {e['f1']:.4f} | {e['roc_auc']:.4f} |")
    lines.append("\n## Observations & Insights\n")
    lines.append("- Random Forest: good accuracy but may under-predict rare attrition class when data is imbalanced.")
    lines.append("- SVM variants trained with class_weight='balanced' typically improved recall (catching Attrition=Yes).")
    if rf_feature_importances is not None:
        lines.append("\n## Top features from Random Forest\n")
        for feat, val in rf_feature_importances.items():
            lines.append(f"- {feat}: {val:.4f}")
    lines.append("\n## Recommendations\n")
    lines.append("- Address class imbalance (SMOTE / resampling).")
    lines.append("- Threshold tuning for probability-based models.")
    lines.append("- Try ensemble/stacking and add explainability (SHAP).")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Report written to", REPORT_MD)

# ---------- Main ----------
def main():
    print("=== Employee Attrition: single-file pipeline ===")
    # 1. Preprocess
    X_train, X_test, y_train, y_test = preprocess()

    # 2. Random Forest
    rf = train_random_forest(X_train, y_train)
    rf_eval = evaluate_model(rf, X_test, y_test, "rf")
    # feature importances
    feat_names = X_train.columns.tolist()
    importances = rf.feature_importances_
    feat_imp = dict(zip(feat_names, importances))
    # sort and keep top 8
    feat_top = dict(sorted(feat_imp.items(), key=lambda kv: kv[1], reverse=True)[:8])

    # 3. SVMs
    svms = train_svms(X_train, y_train)
    svm_results = []
    svm_results.append(evaluate_model(svms["linear"], X_test, y_test, "svm_linear"))
    svm_results.append(evaluate_model(svms["poly"], X_test, y_test, "svm_poly"))
    svm_results.append(evaluate_model(svms["rbf"], X_test, y_test, "svm_rbf"))

    # 4. Build combined results and report
    combined = [rf_eval] + svm_results
    # add classification reports for convenience (printed)
    print("\n--- Classification report: Random Forest ---")
    print(classification_report(y_test, rf.predict(X_test), zero_division=0))
    for name, model in svms.items():
        print(f"\n--- Classification report: SVM ({name}) ---")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))

    build_report(combined, rf_feature_importances=feat_top)
    print("\nAll done. Files saved in current folder. To push to GitHub, add these files to a repo.")
    print("If you want to remove venv and models, use your OS or the cleanup commands.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("Error:", str(e))
        raise
