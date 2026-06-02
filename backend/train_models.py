"""
tumorsense — model training script
Trains one SVM per kernel (rbf, linear, poly, sigmoid) using the best
hyperparameters found via GridSearchCV, then saves each as a joblib bundle
alongside the shared StandardScaler.

Output files (in ./models/):
  scaler.joblib
  model_rbf.joblib
  model_linear.joblib
  model_poly.joblib
  model_sigmoid.joblib
  metrics.json
"""

import json
import os
import warnings

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

data = load_breast_cancer()
X, y = data.data, data.target          # 569 samples, 30 features
FEATURE_NAMES = list(data.feature_names)
TARGET_NAMES  = list(data.target_names)   # ['malignant', 'benign']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")
print("✓ scaler saved")

KERNELS = {
    "rbf": {
        "C":     [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    "linear": {
        "C": [0.01, 0.1, 1, 10, 100],
    },
    "poly": {
        "C":      [0.1, 1, 10],
        "degree": [2, 3, 4],
        "gamma":  ["scale", "auto"],
        "coef0":  [0.0, 1.0],
    },
    "sigmoid": {
        "C":     [0.1, 1, 10],
        "gamma": ["scale", "auto", 0.001, 0.01],
        "coef0": [0.0, 0.5, 1.0],
    },
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_metrics = {}

for kernel, param_grid in KERNELS.items():
    print(f"\nkernel={kernel}")

    gs = GridSearchCV(
        SVC(kernel=kernel, probability=True, random_state=42),
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train_s, y_train)

    best = gs.best_estimator_
    print(f"  best params : {gs.best_params_}")
    print(f"  CV AUC      : {gs.best_score_:.4f}")

    y_pred  = best.predict(X_test_s)
    y_proba = best.predict_proba(X_test_s)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(
        y_test, y_pred, target_names=TARGET_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  test AUC    : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    bundle = {
        "model":        best,
        "kernel":       kernel,
        "best_params":  gs.best_params_,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
    }
    path = f"models/model_{kernel}.joblib"
    joblib.dump(bundle, path)
    print(f"saved {path}")

    all_metrics[kernel] = {
        "best_params":  gs.best_params_,
        "cv_auc":       round(gs.best_score_, 4),
        "test_auc":     round(auc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }

with open("models/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print("\nmetrics.json saved")
print("\nmodels/directory ready for the Flask server.")