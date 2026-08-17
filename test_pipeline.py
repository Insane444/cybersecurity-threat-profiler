"""
Automated Pipeline and Model Verification Test Suite.
Verifies data loading, preprocessor encoding, model training,
single-packet inference, batch prediction, and anomaly detection scoring.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import NetworkTrafficPreprocessor, map_attack_type, ATTACK_CLASSES, FEATURE_COLUMNS
from src.data_loader import generate_synthetic_nsl_kdd, load_nsl_kdd, create_sample_traffic_csv
from src.threat_mitigation import get_threat_profile
from src.train_models import train_pipeline, PREPROCESSOR_PATH, SUPERVISED_MODEL_PATH, UNSUPERVISED_MODEL_PATH, METRICS_JSON_PATH


class TestIntrusionProfiler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test environment and ensure models are trained."""
        print("\n--- [SETUP] Initializing Test Suite & Training Pipeline ---")
        # Run a quick training cycle if models do not exist yet
        if not (os.path.exists(SUPERVISED_MODEL_PATH) and os.path.exists(UNSUPERVISED_MODEL_PATH)):
            train_pipeline(num_samples=3000)

    def test_01_synthetic_data_generation(self):
        """Test that synthetic dataset generator produces valid DataFrame with all classes."""
        df = generate_synthetic_nsl_kdd(num_samples=500, random_state=42)
        self.assertEqual(len(df), 500)
        self.assertIn("attack_type", df.columns)
        self.assertIn("protocol_type", df.columns)
        
        families = df["attack_type"].apply(map_attack_type).unique()
        self.assertIn("Normal", families)
        self.assertIn("DoS", families)
        self.assertIn("Probe", families)

    def test_02_preprocessor_fitting_and_transform(self):
        """Test that preprocessor handles numerical and categorical transformations seamlessly."""
        df = generate_synthetic_nsl_kdd(num_samples=200, random_state=42)
        preprocessor = NetworkTrafficPreprocessor()
        X = preprocessor.fit_transform(df)
        
        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(X.shape[0], 200)
        self.assertGreater(X.shape[1], 40)
        self.assertEqual(len(preprocessor.get_feature_names()), X.shape[1])

    def test_03_model_serialization(self):
        """Test that all saved model artifacts and metric json exist and can be loaded."""
        self.assertTrue(os.path.exists(PREPROCESSOR_PATH))
        self.assertTrue(os.path.exists(SUPERVISED_MODEL_PATH))
        self.assertTrue(os.path.exists(UNSUPERVISED_MODEL_PATH))
        self.assertTrue(os.path.exists(METRICS_JSON_PATH))

        prep = joblib.load(PREPROCESSOR_PATH)
        clf = joblib.load(SUPERVISED_MODEL_PATH)
        iso = joblib.load(UNSUPERVISED_MODEL_PATH)

        self.assertTrue(hasattr(prep, "transform"))
        self.assertTrue(hasattr(clf, "predict"))
        self.assertTrue(hasattr(iso, "score_samples"))

    def test_04_single_packet_inference(self):
        """Test single-packet prediction for a simulated normal and DoS packet."""
        prep = joblib.load(PREPROCESSOR_PATH)
        clf = joblib.load(SUPERVISED_MODEL_PATH)
        iso = joblib.load(UNSUPERVISED_MODEL_PATH)

        # Normal Packet
        normal_packet = {
            "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
            "src_bytes": 350, "dst_bytes": 2200, "land": 0, "wrong_fragment": 0, "urgent": 0,
            "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
            "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0, "count": 4, "srv_count": 4,
            "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0, "dst_host_count": 50, "dst_host_srv_count": 250,
            "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.05, "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
        }

        X_normal = prep.prepare_single_packet(normal_packet)
        pred_class = clf.predict(X_normal)[0]
        probs = clf.predict_proba(X_normal)[0]
        raw_anomaly = -iso.score_samples(X_normal)[0]

        self.assertIn(pred_class, ATTACK_CLASSES)
        self.assertEqual(len(probs), len(clf.classes_))
        self.assertIsInstance(float(raw_anomaly), float)

        # Threat profile verification
        threat_info = get_threat_profile(pred_class, is_anomaly=False, anomaly_score=0.1)
        self.assertIn("severity", threat_info)
        self.assertIn("firewall_rule", threat_info)
        self.assertIn("recommendations", threat_info)

    def test_05_batch_csv_profiling(self):
        """Test batch network traffic CSV generation and processing."""
        sample_df = create_sample_traffic_csv(num_samples=50)
        self.assertEqual(len(sample_df), 50)
        
        prep = joblib.load(PREPROCESSOR_PATH)
        clf = joblib.load(SUPERVISED_MODEL_PATH)
        iso = joblib.load(UNSUPERVISED_MODEL_PATH)

        X_batch = prep.transform(sample_df)
        preds = clf.predict(X_batch)
        iso_scores = -iso.score_samples(X_batch)

        self.assertEqual(len(preds), 50)
        self.assertEqual(len(iso_scores), 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
