"""
Dataset Loading and Realistic Synthetic Generation for NSL-KDD.
Handles automatic fetching from remote NSL-KDD mirrors, local file parsing,
and robust synthetic traffic generation if network/files are unavailable.
"""

import os
import sys
import urllib.request
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import NSL_KDD_COLUMNS, FEATURE_COLUMNS, map_attack_type, TOP_SERVICES, TOP_FLAGS

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
KDD_TRAIN_FILE = os.path.join(DATASET_DIR, "KDDTrain+.txt")
KDD_TEST_FILE = os.path.join(DATASET_DIR, "KDDTest+.txt")
SAMPLE_CSV_FILE = os.path.join(DATASET_DIR, "sample_traffic.csv")

REMOTE_URLS = [
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "https://raw.githubusercontent.com/jtkill/NSL-KDD-Dataset/master/KDDTrain+.txt",
    "https://raw.githubusercontent.com/cyber-threat-intelligence/NSL-KDD/main/KDDTrain+.txt"
]


def ensure_dataset_dir():
    """Ensures the dataset directory exists."""
    os.makedirs(DATASET_DIR, exist_ok=True)


def download_nsl_kdd(dest_path: str = KDD_TRAIN_FILE) -> bool:
    """Attempts to download the official NSL-KDD train dataset from remote mirrors."""
    ensure_dataset_dir()
    for url in REMOTE_URLS:
        try:
            print(f"[INFO] Attempting to download NSL-KDD from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
                print(f"[SUCCESS] Downloaded NSL-KDD dataset to {dest_path}")
                return True
        except Exception as e:
            print(f"[WARN] Failed downloading from {url}: {e}")
    return False


def generate_synthetic_nsl_kdd(num_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a statistically realistic synthetic NSL-KDD network dataset with
    accurate feature distributions across Normal, DoS, Probe, R2L, and U2R traffic.
    """
    np.random.seed(random_state)
    ensure_dataset_dir()

    # Define distribution of classes (realistic network ratio)
    # 55% Normal, 28% DoS, 12% Probe, 4% R2L, 1% U2R
    proportions = {"Normal": 0.55, "DoS": 0.28, "Probe": 0.12, "R2L": 0.04, "U2R": 0.01}
    class_counts = {cls: int(num_samples * prop) for cls, prop in proportions.items()}
    # Adjust last class for total match
    class_counts["Normal"] += num_samples - sum(class_counts.values())

    rows = []

    for attack_class, count in class_counts.items():
        for _ in range(count):
            row = {}
            if attack_class == "Normal":
                row["duration"] = int(np.random.exponential(scale=2))
                row["protocol_type"] = np.random.choice(["tcp", "udp", "icmp"], p=[0.75, 0.20, 0.05])
                row["service"] = np.random.choice(["http", "smtp", "domain_u", "ftp_data", "auth", "private"], p=[0.55, 0.15, 0.15, 0.08, 0.04, 0.03])
                row["flag"] = np.random.choice(["SF", "S0", "REJ"], p=[0.94, 0.04, 0.02])
                row["src_bytes"] = int(np.random.exponential(scale=650) + 150)
                row["dst_bytes"] = int(np.random.exponential(scale=2500) + 300)
                row["land"] = 0
                row["wrong_fragment"] = 0
                row["urgent"] = 0
                row["hot"] = 0
                row["num_failed_logins"] = 0
                row["logged_in"] = 1 if row["protocol_type"] == "tcp" and row["service"] in ["http", "smtp", "ftp_data"] else 0
                row["num_compromised"] = 0
                row["root_shell"] = 0
                row["su_attempted"] = 0
                row["num_root"] = 0
                row["num_file_creations"] = 0
                row["num_shells"] = 0
                row["num_access_files"] = 0
                row["num_outbound_cmds"] = 0
                row["is_host_login"] = 0
                row["is_guest_login"] = 0
                row["count"] = int(np.random.randint(1, 15))
                row["srv_count"] = int(np.random.randint(1, 15))
                row["serror_rate"] = float(np.random.beta(0.5, 20))
                row["srv_serror_rate"] = float(np.random.beta(0.5, 20))
                row["rerror_rate"] = float(np.random.beta(0.5, 30))
                row["srv_rerror_rate"] = float(np.random.beta(0.5, 30))
                row["same_srv_rate"] = float(np.random.uniform(0.85, 1.0))
                row["diff_srv_rate"] = float(np.random.uniform(0.0, 0.15))
                row["srv_diff_host_rate"] = float(np.random.uniform(0.0, 0.2))
                row["dst_host_count"] = int(np.random.randint(1, 255))
                row["dst_host_srv_count"] = int(np.random.randint(1, 255))
                row["dst_host_same_srv_rate"] = float(np.random.uniform(0.7, 1.0))
                row["dst_host_diff_srv_rate"] = float(np.random.uniform(0.0, 0.1))
                row["dst_host_same_src_port_rate"] = float(np.random.uniform(0.0, 0.3))
                row["dst_host_srv_diff_host_rate"] = float(np.random.uniform(0.0, 0.1))
                row["dst_host_serror_rate"] = float(np.random.uniform(0.0, 0.05))
                row["dst_host_srv_serror_rate"] = float(np.random.uniform(0.0, 0.05))
                row["dst_host_rerror_rate"] = float(np.random.uniform(0.0, 0.05))
                row["dst_host_srv_rerror_rate"] = float(np.random.uniform(0.0, 0.05))
                row["attack_type"] = "normal"
                row["level"] = 21

            elif attack_class == "DoS":
                # High traffic volume, SYN floods, S0 flags, high count & error rates
                row["duration"] = 0
                row["protocol_type"] = np.random.choice(["tcp", "icmp", "udp"], p=[0.70, 0.25, 0.05])
                row["service"] = np.random.choice(["private", "http", "eco_i", "ecr_i", "finger", "other"], p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05])
                row["flag"] = np.random.choice(["S0", "REJ", "SF"], p=[0.75, 0.20, 0.05])
                row["src_bytes"] = int(np.random.choice([0, 0, 42, 1032, 1460]))
                row["dst_bytes"] = 0
                row["land"] = 1 if np.random.rand() < 0.05 else 0
                row["wrong_fragment"] = int(np.random.choice([0, 1, 3], p=[0.9, 0.08, 0.02]))
                row["urgent"] = 0
                row["hot"] = 0
                row["num_failed_logins"] = 0
                row["logged_in"] = 0
                row["num_compromised"] = 0
                row["root_shell"] = 0
                row["su_attempted"] = 0
                row["num_root"] = 0
                row["num_file_creations"] = 0
                row["num_shells"] = 0
                row["num_access_files"] = 0
                row["num_outbound_cmds"] = 0
                row["is_host_login"] = 0
                row["is_guest_login"] = 0
                row["count"] = int(np.random.randint(150, 511))
                row["srv_count"] = int(np.random.randint(10, 511))
                row["serror_rate"] = float(np.random.uniform(0.8, 1.0)) if row["flag"] == "S0" else 0.0
                row["srv_serror_rate"] = row["serror_rate"]
                row["rerror_rate"] = float(np.random.uniform(0.8, 1.0)) if row["flag"] == "REJ" else 0.0
                row["srv_rerror_rate"] = row["rerror_rate"]
                row["same_srv_rate"] = float(np.random.uniform(0.9, 1.0))
                row["diff_srv_rate"] = float(np.random.uniform(0.0, 0.1))
                row["srv_diff_host_rate"] = 0.0
                row["dst_host_count"] = 255
                row["dst_host_srv_count"] = int(np.random.randint(1, 50))
                row["dst_host_same_srv_rate"] = float(np.random.uniform(0.01, 0.2))
                row["dst_host_diff_srv_rate"] = float(np.random.uniform(0.05, 0.4))
                row["dst_host_same_src_port_rate"] = float(np.random.uniform(0.0, 0.2))
                row["dst_host_srv_diff_host_rate"] = 0.0
                row["dst_host_serror_rate"] = row["serror_rate"]
                row["dst_host_srv_serror_rate"] = row["srv_serror_rate"]
                row["dst_host_rerror_rate"] = row["rerror_rate"]
                row["dst_host_srv_rerror_rate"] = row["srv_rerror_rate"]
                row["attack_type"] = np.random.choice(["neptune", "smurf", "teardrop", "back", "pod"])
                row["level"] = int(np.random.randint(1, 15))

            elif attack_class == "Probe":
                # Port scanning / network sweeps, REJ / RSTO flags, high diff_srv_rate
                row["duration"] = int(np.random.randint(0, 3))
                row["protocol_type"] = np.random.choice(["tcp", "icmp", "udp"], p=[0.6, 0.3, 0.1])
                row["service"] = np.random.choice(["private", "eco_i", "telnet", "finger", "ftp", "other"], p=[0.35, 0.25, 0.15, 0.1, 0.1, 0.05])
                row["flag"] = np.random.choice(["REJ", "RSTO", "SF", "S0"], p=[0.45, 0.30, 0.15, 0.10])
                row["src_bytes"] = int(np.random.choice([0, 8, 44, 128]))
                row["dst_bytes"] = int(np.random.choice([0, 0, 44, 60]))
                row["land"] = 0
                row["wrong_fragment"] = 0
                row["urgent"] = 0
                row["hot"] = 0
                row["num_failed_logins"] = 0
                row["logged_in"] = 0
                row["num_compromised"] = 0
                row["root_shell"] = 0
                row["su_attempted"] = 0
                row["num_root"] = 0
                row["num_file_creations"] = 0
                row["num_shells"] = 0
                row["num_access_files"] = 0
                row["num_outbound_cmds"] = 0
                row["is_host_login"] = 0
                row["is_guest_login"] = 0
                row["count"] = int(np.random.randint(1, 100))
                row["srv_count"] = int(np.random.randint(1, 20))
                row["serror_rate"] = 0.0
                row["srv_serror_rate"] = 0.0
                row["rerror_rate"] = float(np.random.uniform(0.4, 0.9))
                row["srv_rerror_rate"] = float(np.random.uniform(0.4, 0.9))
                row["same_srv_rate"] = float(np.random.uniform(0.05, 0.3))
                row["diff_srv_rate"] = float(np.random.uniform(0.6, 1.0))
                row["srv_diff_host_rate"] = float(np.random.uniform(0.3, 0.9))
                row["dst_host_count"] = 255
                row["dst_host_srv_count"] = int(np.random.randint(1, 30))
                row["dst_host_same_srv_rate"] = float(np.random.uniform(0.01, 0.2))
                row["dst_host_diff_srv_rate"] = float(np.random.uniform(0.5, 1.0))
                row["dst_host_same_src_port_rate"] = float(np.random.uniform(0.5, 1.0))
                row["dst_host_srv_diff_host_rate"] = float(np.random.uniform(0.3, 0.8))
                row["dst_host_serror_rate"] = 0.0
                row["dst_host_srv_serror_rate"] = 0.0
                row["dst_host_rerror_rate"] = float(np.random.uniform(0.4, 0.9))
                row["dst_host_srv_rerror_rate"] = float(np.random.uniform(0.4, 0.9))
                row["attack_type"] = np.random.choice(["ipsweep", "portsweep", "nmap", "satan"])
                row["level"] = int(np.random.randint(5, 18))

            elif attack_class == "R2L":
                # Remote login attempts, guest login, failed logins, hot commands
                row["duration"] = int(np.random.exponential(scale=10) + 1)
                row["protocol_type"] = "tcp"
                row["service"] = np.random.choice(["ftp", "telnet", "smtp", "imap", "pop_3"], p=[0.4, 0.3, 0.15, 0.1, 0.05])
                row["flag"] = "SF"
                row["src_bytes"] = int(np.random.randint(200, 3000))
                row["dst_bytes"] = int(np.random.randint(500, 8000))
                row["land"] = 0
                row["wrong_fragment"] = 0
                row["urgent"] = 0
                row["hot"] = int(np.random.randint(1, 8))
                row["num_failed_logins"] = int(np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05]))
                row["logged_in"] = int(np.random.choice([0, 1], p=[0.7, 0.3]))
                row["num_compromised"] = int(np.random.randint(0, 3))
                row["root_shell"] = 0
                row["su_attempted"] = 0
                row["num_root"] = 0
                row["num_file_creations"] = int(np.random.randint(0, 2))
                row["num_shells"] = 0
                row["num_access_files"] = 0
                row["num_outbound_cmds"] = 0
                row["is_host_login"] = 0
                row["is_guest_login"] = int(np.random.choice([0, 1], p=[0.5, 0.5]))
                row["count"] = int(np.random.randint(1, 10))
                row["srv_count"] = int(np.random.randint(1, 10))
                row["serror_rate"] = 0.0
                row["srv_serror_rate"] = 0.0
                row["rerror_rate"] = 0.0
                row["srv_rerror_rate"] = 0.0
                row["same_srv_rate"] = 1.0
                row["diff_srv_rate"] = 0.0
                row["srv_diff_host_rate"] = 0.0
                row["dst_host_count"] = int(np.random.randint(1, 80))
                row["dst_host_srv_count"] = int(np.random.randint(1, 80))
                row["dst_host_same_srv_rate"] = 1.0
                row["dst_host_diff_srv_rate"] = 0.0
                row["dst_host_same_src_port_rate"] = float(np.random.uniform(0.0, 0.2))
                row["dst_host_srv_diff_host_rate"] = 0.0
                row["dst_host_serror_rate"] = 0.0
                row["dst_host_srv_serror_rate"] = 0.0
                row["dst_host_rerror_rate"] = 0.0
                row["dst_host_srv_rerror_rate"] = 0.0
                row["attack_type"] = np.random.choice(["warezclient", "guess_passwd", "ftp_write", "imap", "phf"])
                row["level"] = int(np.random.randint(1, 12))

            else:  # U2R
                # User escalation to root, root_shell, file creations, su attempts
                row["duration"] = int(np.random.randint(10, 120))
                row["protocol_type"] = "tcp"
                row["service"] = np.random.choice(["telnet", "ftp_data", "other"], p=[0.6, 0.3, 0.1])
                row["flag"] = "SF"
                row["src_bytes"] = int(np.random.randint(500, 4000))
                row["dst_bytes"] = int(np.random.randint(1000, 15000))
                row["land"] = 0
                row["wrong_fragment"] = 0
                row["urgent"] = int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
                row["hot"] = int(np.random.randint(2, 10))
                row["num_failed_logins"] = 0
                row["logged_in"] = 1
                row["num_compromised"] = int(np.random.randint(1, 10))
                row["root_shell"] = int(np.random.choice([1, 1], p=[0.5, 0.5]))
                row["su_attempted"] = int(np.random.choice([1, 2], p=[0.7, 0.3]))
                row["num_root"] = int(np.random.randint(1, 15))
                row["num_file_creations"] = int(np.random.randint(1, 6))
                row["num_shells"] = int(np.random.choice([1, 2], p=[0.8, 0.2]))
                row["num_access_files"] = int(np.random.randint(1, 4))
                row["num_outbound_cmds"] = 0
                row["is_host_login"] = 0
                row["is_guest_login"] = 0
                row["count"] = int(np.random.randint(1, 5))
                row["srv_count"] = int(np.random.randint(1, 5))
                row["serror_rate"] = 0.0
                row["srv_serror_rate"] = 0.0
                row["rerror_rate"] = 0.0
                row["srv_rerror_rate"] = 0.0
                row["same_srv_rate"] = 1.0
                row["diff_srv_rate"] = 0.0
                row["srv_diff_host_rate"] = 0.0
                row["dst_host_count"] = int(np.random.randint(1, 50))
                row["dst_host_srv_count"] = int(np.random.randint(1, 50))
                row["dst_host_same_srv_rate"] = 1.0
                row["dst_host_diff_srv_rate"] = 0.0
                row["dst_host_same_src_port_rate"] = float(np.random.uniform(0.0, 0.1))
                row["dst_host_srv_diff_host_rate"] = 0.0
                row["dst_host_serror_rate"] = 0.0
                row["dst_host_srv_serror_rate"] = 0.0
                row["dst_host_rerror_rate"] = 0.0
                row["dst_host_srv_rerror_rate"] = 0.0
                row["attack_type"] = np.random.choice(["buffer_overflow", "rootkit", "loadmodule", "perl"])
                row["level"] = int(np.random.randint(1, 10))

            rows.append(row)

    df = pd.DataFrame(rows)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def load_nsl_kdd(filepath: str = None, fallback_synthetic_samples: int = 20000) -> pd.DataFrame:
    """
    Loads NSL-KDD dataset from file if available; otherwise tries downloading,
    or falls back to generating a realistic synthetic dataset.
    """
    ensure_dataset_dir()
    target_path = filepath or KDD_TRAIN_FILE

    # 1. Check if local file exists
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[INFO] Loading local dataset from {target_path}")
        try:
            df = pd.read_csv(target_path, header=None)
            if df.shape[1] == len(NSL_KDD_COLUMNS):
                df.columns = NSL_KDD_COLUMNS
            elif df.shape[1] == len(NSL_KDD_COLUMNS) - 1:
                df.columns = [c for c in NSL_KDD_COLUMNS if c != "level"]
            else:
                df.columns = NSL_KDD_COLUMNS[:df.shape[1]]
            return df
        except Exception as e:
            print(f"[WARN] Error reading local file {target_path}: {e}")

    # 2. Try downloading if default file missing
    if target_path == KDD_TRAIN_FILE and download_nsl_kdd(KDD_TRAIN_FILE):
        try:
            df = pd.read_csv(KDD_TRAIN_FILE, header=None)
            if df.shape[1] == len(NSL_KDD_COLUMNS):
                df.columns = NSL_KDD_COLUMNS
            elif df.shape[1] == len(NSL_KDD_COLUMNS) - 1:
                df.columns = [c for c in NSL_KDD_COLUMNS if c != "level"]
            else:
                df.columns = NSL_KDD_COLUMNS[:df.shape[1]]
            return df
        except Exception as e:
            print(f"[WARN] Error reading downloaded file: {e}")

    # 3. Fallback to generating synthetic dataset
    print(f"[INFO] Generating high-fidelity synthetic NSL-KDD dataset ({fallback_synthetic_samples} samples)...")
    synthetic_df = generate_synthetic_nsl_kdd(num_samples=fallback_synthetic_samples)
    
    # Save a copy to disk for reuse
    synthetic_df.to_csv(KDD_TRAIN_FILE, index=False, header=False)
    print(f"[INFO] Saved synthetic training dataset to {KDD_TRAIN_FILE}")
    
    return synthetic_df


def create_sample_traffic_csv(output_path: str = SAMPLE_CSV_FILE, num_samples: int = 250):
    """Generates a diverse sample batch of network traffic for UI profiling tests."""
    sample_df = generate_synthetic_nsl_kdd(num_samples=num_samples, random_state=99)
    # Add high-level attack family column
    sample_df["attack_family"] = sample_df["attack_type"].apply(map_attack_type)
    sample_df.to_csv(output_path, index=False)
    print(f"[INFO] Sample batch traffic CSV created at: {output_path}")
    return sample_df


if __name__ == "__main__":
    df = load_nsl_kdd()
    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df["attack_type"].apply(map_attack_type).value_counts())
    create_sample_traffic_csv()
