# step6_smote.py
"""
Apply SMOTE to training data, retrain RF and RBF-SVM, evaluate, and save summary.
Run: python step6_smote.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Filenames (must exist)
TRAIN_CSV = "preprocessed_train.csv"
TEST_CSV = "preprocessed_test.csv"
OUT_SUM = "smote_summary.csv"
RF_SMOTE_MODEL = "rf_smote.joblib"
SVM_RBF_SMOTE = "svm_rbf_smote.joblib"

RANDOM_STATE = 42

def load_data():
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        raise FileNotFoundError("Preprocessed files not found. Run preprocessing (step2) or main.py first.")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    X_train = train.drop(columns=["Attrition"])
    y_train = train["Attrition"]
    X_test = test.drop(columns=["Attrition"])
    y_test = test["Attrition"]
    return X_train, X_test, y_train, y_test

def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # get probability-like scores
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        try:
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min())/(scores.max()-scores.min()+1e-9)
        except:
            y_proba = np.zeros_like(y_pred, dtype=float)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test))>1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print("Train shape before SMOTE:", X_train.shape, "positive rate:", y_train.mean())

    # Apply SMOTE (default k_neighbors=5; you can reduce if very few positives)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE, train shape:", X_res.shape, "positive rate:", y_res.mean())

    # 1) Random Forest on SMOTE data
    print("\nTraining Random Forest on SMOTE data...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_res, y_res)
    joblib.dump(rf, RF_SMOTE_MODEL)
    print("Saved", RF_SMOTE_MODEL)
    rf_eval = eval_model(rf, X_test, y_test)
    print("\nRandom Forest (SMOTE) evaluation:")
    print(classification_report(y_test, rf.predict(X_test), zero_division=0))
    print("Confusion Matrix:", rf_eval["confusion_matrix"])

    # 2) RBF SVM on SMOTE data (train with class_weight balanced too)
    print("\nTraining RBF SVM (on SMOTE data) - may take a short while...")
    svm_rbf = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE, C=1.0, gamma="scale")
    svm_rbf.fit(X_res, y_res)
    joblib.dump(svm_rbf, SVM_RBF_SMOTE)
    print("Saved", SVM_RBF_SMOTE)
    svm_eval = eval_model(svm_rbf, X_test, y_test)
    print("\nSVM RBF (SMOTE) evaluation:")
    print(classification_report(y_test, svm_rbf.predict(X_test), zero_division=0))
    print("Confusion Matrix:", svm_eval["confusion_matrix"])

    # Save summary CSV
    rows = []
    rows.append({"model":"RF_SMOTE", **rf_eval})
    rows.append({"model":"SVM_RBF_SMOTE", **svm_eval})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_SUM, index=False)
    print("\nSaved summary to", OUT_SUM)
    print(df.to_string(index=False))

    # Quick guidance: if recall improved, keep SMOTE results in report
    print("\nIf recall improved for RF or SVM, update Model_Comparison_Report.md with the SMOTE results and include a short note about oversampling.")

if __name__ == "__main__":
    main()
