"""
Network Traffic Preprocessing & Feature Engineering Module for NSL-KDD Dataset.
Provides canonical column definitions, attack category mappings, One-Hot Encoding,
standard scaling, and single-packet inference transformations.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Canonical NSL-KDD 41 core feature columns + label + difficulty level (43 columns total)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "level"
]

# Primary feature columns (excluding label and level)
FEATURE_COLUMNS = [col for col in NSL_KDD_COLUMNS if col not in ["attack_type", "level"]]

# Categorical and numerical columns
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERICAL_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]

# Canonical Attack Mapping: Maps granular attack types to 5 high-level classes
ATTACK_MAPPING = {
    # Normal Traffic
    "normal": "Normal",
    
    # Denial of Service (DoS)
    "neptune": "DoS",
    "smurf": "DoS",
    "back": "DoS",
    "teardrop": "DoS",
    "pod": "DoS",
    "land": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    
    # Surveillance & Probing (Probe)
    "ipsweep": "Probe",
    "portsweep": "Probe",
    "nmap": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    
    # Remote to Local (R2L)
    "warezclient": "R2L",
    "guess_passwd": "R2L",
    "warezmaster": "R2L",
    "imap": "R2L",
    "ftp_write": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "worm": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",
    
    # User to Root (U2R)
    "buffer_overflow": "U2R",
    "rootkit": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R"
}

# The 5 core attack classes
ATTACK_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Top common protocols, services, and flags for quick validation & UI selectors
PROTOCOLS = ["tcp", "udp", "icmp"]

TOP_SERVICES = [
    "http", "smtp", "ftp", "ftp_data", "domain_u", "telnet", "finger", 
    "eco_i", "ecr_i", "auth", "private", "pop_3", "other", "time", 
    "ssh", "dns", "ntp", "sunrpc", "urp_i", "courier", "daytime"
]

TOP_FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTR", "S1", "S2", "S3", "OTH", "SHR", "RSTOS0"]


def map_attack_type(raw_label: str) -> str:
    """Maps a raw NSL-KDD attack label to one of the 5 canonical classes."""
    if not isinstance(raw_label, str):
        return "Normal"
    clean_label = str(raw_label).strip().lower().rstrip(".")
    return ATTACK_MAPPING.get(clean_label, "Normal" if clean_label == "normal" else "DoS")


class NetworkTrafficPreprocessor:
    """
    Handles end-to-end preprocessing, feature scaling, and one-hot encoding
    for NSL-KDD network intrusion data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        self.feature_names_ = []
        self.numerical_cols = NUMERICAL_FEATURES
        self.categorical_cols = CATEGORICAL_FEATURES

    def fit(self, df: pd.DataFrame):
        """Fits scaler on numerical columns and OneHotEncoder on categorical columns."""
        clean_df = df.copy()
        
        # Ensure all expected columns exist
        for col in self.numerical_cols:
            if col not in clean_df.columns:
                clean_df[col] = 0.0
            else:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").fillna(0.0)

        for col in self.categorical_cols:
            if col not in clean_df.columns:
                clean_df[col] = "other"
            else:
                clean_df[col] = clean_df[col].astype(str).str.lower().str.strip()

        # Fit transformers
        self.scaler.fit(clean_df[self.numerical_cols])
        self.encoder.fit(clean_df[self.categorical_cols])
        
        # Build feature names
        cat_feature_names = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names_ = self.numerical_cols + cat_feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms raw network traffic DataFrame into preprocessed feature matrix."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
        
        clean_df = df.copy()
        for col in self.numerical_cols:
            if col not in clean_df.columns:
                clean_df[col] = 0.0
            else:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").fillna(0.0)

        for col in self.categorical_cols:
            if col not in clean_df.columns:
                clean_df[col] = "other"
            else:
                clean_df[col] = clean_df[col].astype(str).str.lower().str.strip()

        num_scaled = self.scaler.transform(clean_df[self.numerical_cols])
        cat_encoded = self.encoder.transform(clean_df[self.categorical_cols])

        return np.hstack([num_scaled, cat_encoded])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fits on df and returns transformed feature matrix."""
        return self.fit(df).transform(df)

    def prepare_single_packet(self, packet_dict: dict) -> np.ndarray:
        """Converts a single packet dictionary to a transformed 2D feature matrix (1, N)."""
        single_df = pd.DataFrame([packet_dict])
        return self.transform(single_df)

    def get_feature_names(self) -> list:
        """Returns the list of transformed feature names."""
        return self.feature_names_

    def explain_single_packet_contributions(self, packet_dict: dict, global_importances: dict = None) -> list:
        """
        Computes local feature contribution deviations relative to standard benign profile.
        Returns sorted list of top feature drivers for Explainable AI (XAI) UI charts.
        """
        if not self.is_fitted:
            return []
        
        X_trans = self.prepare_single_packet(packet_dict)[0]
        contributions = []

        for i, feat_name in enumerate(self.feature_names_):
            val = float(X_trans[i])
            # Weight by global feature importance if available, or by magnitude
            weight = global_importances.get(feat_name, 0.05) if global_importances else 0.05
            impact = abs(val) * (1.0 + weight * 5.0)
            contributions.append({
                "feature": feat_name,
                "scaled_value": round(val, 3),
                "impact_score": round(float(impact), 4),
                "direction": "Elevated / Abnormal" if val > 0 else "Suppressed / Below Mean"
            })

        # Sort by impact score descending
        contributions.sort(key=lambda x: x["impact_score"], reverse=True)
        return contributions[:10]
