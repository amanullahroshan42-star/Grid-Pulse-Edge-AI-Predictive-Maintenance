"""
train_lab_model.py
Trains and evaluates an Edge Machine Learning Model on the curated laboratory dataset:
motor_training_features_dataset.csv
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(CURRENT_DIR, "motor_training_features_dataset.csv")
MODEL_OUT = os.path.join(CURRENT_DIR, "motor_lab_model.joblib")
META_OUT = os.path.join(CURRENT_DIR, "motor_lab_metadata.json")

def train():
    print(f"[*] Loading feature dataset: {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)

    ignore_cols = ["operating_condition", "health_status", "load_kg"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols].values
    y = df["operating_condition"].values
    classes = sorted(list(set(y)))

    print(f"[+] Loaded {len(X)} samples with {len(feature_cols)} features.")
    print("Features:", feature_cols)
    print("Target Classes:", classes)

    # Train / Test split (80 / 20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=40,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("\n[*] Training Random Forest Classifier...")
    clf.fit(X_train, y_train)

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"[+] 5-Fold Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"[+] Holdout Test Accuracy: {test_acc*100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Top feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 6 Most Discriminating Physical Features:")
    for i in range(min(6, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

    joblib.dump(clf, MODEL_OUT, compress=3)
    print(f"\n[+] Saved trained model to: {MODEL_OUT}")

    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 40,
        "features": feature_cols,
        "classes": classes,
        "test_accuracy": float(test_acc),
        "asset": "PSG 2.2 kW Three-Phase Induction Motor (Lab Benchmark)"
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved model metadata to: {META_OUT}")

if __name__ == "__main__":
    train()
