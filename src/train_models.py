"""
ML Training Pipeline: Supervised Threat Classifier & Unsupervised Zero-Day Anomaly Detector.
Trains models on NSL-KDD data, evaluates metrics, and exports serialized model artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc

from src.preprocessor import NetworkTrafficPreprocessor, map_attack_type, ATTACK_CLASSES
from src.data_loader import load_nsl_kdd, create_sample_traffic_csv

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
SUPERVISED_MODEL_PATH = os.path.join(MODELS_DIR, "known_threat_model.pkl")
UNSUPERVISED_MODEL_PATH = os.path.join(MODELS_DIR, "zeroday_model.pkl")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "metrics_summary.json")


def ensure_models_dir():
    """Ensures the models directory exists."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def train_pipeline(data_path: str = None, num_samples: int = 25000):
    """
    Executes the complete data loading, preprocessing, dual-model training,
    evaluation, and artifact serialization pipeline.
    """
    ensure_models_dir()
    print("================================================================")
    print("  CYBERSECURITY INTRUSION PROFILER - MODEL TRAINING PIPELINE    ")
    print("================================================================")

    # 1. Load Data
    print("\n[STEP 1/5] Loading NSL-KDD Dataset...")
    df = load_nsl_kdd(filepath=data_path, fallback_synthetic_samples=num_samples)
    print(f"Dataset Loaded: {df.shape[0]} packets, {df.shape[1]} raw attributes.")

    # 2. Map high-level attack families
    df["attack_family"] = df["attack_type"].apply(map_attack_type)
    print("Attack Distribution:\n", df["attack_family"].value_counts())

    # 3. Preprocessing & Feature Engineering
    print("\n[STEP 2/5] Fitting Preprocessor & Scaling Features...")
    preprocessor = NetworkTrafficPreprocessor()
    X = preprocessor.fit_transform(df)
    y = np.array(df["attack_family"].tolist(), dtype=object)
    feature_names = preprocessor.get_feature_names()
    print(f"Engineered Feature Matrix Shape: {X.shape} (Features: {len(feature_names)})")

    # Split into train / test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 4. Train Engine 1: Known Threat Classifier (Supervised)
    print("\n[STEP 3/5] Training Supervised Known Threat Classifier (Random Forest Ensemble)...")
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Supervised Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Supervised Classification Accuracy: {acc * 100:.2f}%")

    labels_in_test = [cls for cls in ATTACK_CLASSES if cls in y_test]
    cm = confusion_matrix(y_test, y_pred, labels=labels_in_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("\nClassification Report Summary:")
    for cls in labels_in_test:
        cls_rep = report.get(cls, {})
        print(f" - {cls:8s} -> Precision: {cls_rep.get('precision', 0):.3f} | Recall: {cls_rep.get('recall', 0):.3f} | F1: {cls_rep.get('f1-score', 0):.3f}")

    # Extract Feature Importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:20]
    top_features = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in sorted_idx
    ]

    # 5. Train Engine 2: Zero-Day Anomaly Detector (Unsupervised Isolation Forest)
    print("\n[STEP 4/5] Training Unsupervised Zero-Day Anomaly Engine (Isolation Forest on Benign Traffic)...")
    # Train strictly on normal benign baseline traffic
    normal_mask_train = (y_train == "Normal")
    X_train_normal = X_train[normal_mask_train]

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.04,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_normal)

    # Anomaly testing on normal vs attacks
    iso_scores_normal = -iso_forest.score_samples(X_test[y_test == "Normal"])
    iso_scores_attacks = -iso_forest.score_samples(X_test[y_test != "Normal"]) if (y_test != "Normal").any() else np.array([0.5])
    print(f"Mean Anomaly Score (Normal): {np.mean(iso_scores_normal):.4f}")
    print(f"Mean Anomaly Score (Attacks): {np.mean(iso_scores_attacks):.4f}")

    # 6. Save Model Artifacts
    print("\n[STEP 5/5] Exporting Serialized Model Artifacts...")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(clf, SUPERVISED_MODEL_PATH)
    joblib.dump(iso_forest, UNSUPERVISED_MODEL_PATH)

    # Compute ROC Curves and AUC for each class
    roc_data = {}
    try:
        y_test_bin = pd.get_dummies(y_test).reindex(columns=clf.classes_, fill_value=0).values
        y_score = clf.predict_proba(X_test)
        for i, cls in enumerate(clf.classes_):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            # Subsample curve points for lightweight JSON serialization
            step = max(1, len(fpr) // 30)
            roc_data[cls] = {
                "fpr": [round(float(x), 4) for x in fpr[::step]],
                "tpr": [round(float(x), 4) for x in tpr[::step]],
                "auc": round(float(auc(fpr, tpr)), 4)
            }
    except Exception as e:
        print(f"[WARN] Error calculating ROC curves: {e}")

    metrics_summary = {
        "accuracy": float(acc),
        "total_training_samples": int(X_train.shape[0]),
        "total_test_samples": int(X_test.shape[0]),
        "total_features": int(len(feature_names)),
        "attack_classes": ATTACK_CLASSES,
        "labels_in_test": labels_in_test,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "top_features": top_features,
        "roc_data": roc_data,
        "anomaly_baseline_mean": float(np.mean(iso_scores_normal)),
        "anomaly_baseline_std": float(np.std(iso_scores_normal)),
        "anomaly_threshold": float(np.percentile(iso_scores_normal, 95))
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Generate sample testing CSV
    create_sample_traffic_csv()

    print(f"[SUCCESS] Saved Preprocessor: {PREPROCESSOR_PATH}")
    print(f"[SUCCESS] Saved Known Threat Model: {SUPERVISED_MODEL_PATH}")
    print(f"[SUCCESS] Saved Zero-Day Anomaly Model: {UNSUPERVISED_MODEL_PATH}")
    print(f"[SUCCESS] Saved Metrics Summary: {METRICS_JSON_PATH}")
    print("================================================================\n")
    return metrics_summary


if __name__ == "__main__":
    train_pipeline()
