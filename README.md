# 🛡️ Cybersecurity Network Threat & Intrusion Profiler

> **A Production-Ready Full-Stack Security Web Application & ML Inference Engine powered by Streamlit and the NSL-KDD Network Intrusion Dataset.**

---

## 📌 Executive Summary

Modern computer networks face both known cyber threats and newly invented attack methods. This system integrates:
1. **Supervised Multi-Class Classifier (Random Forest / XGBoost):** Accurately categorizes network traffic into 5 canonical families (`Normal`, `DoS`, `Probe`, `R2L`, `U2R`).
2. **Unsupervised Outlier & Anomaly Engine (Isolation Forest):** Establishes an empirical baseline from benign traffic to detect novel, unseen **Zero-Day network intrusions**.
3. **Interactive SOC Dashboard (Streamlit & Plotly):** Provides real-time packet inspection, automated MITRE ATT&CK mapping, and copyable Linux `iptables` / Snort playbooks.

---

## 🚀 Key Features

- **📊 Executive Threat Dashboard:** High-level KPIs, threat distribution donut charts, isolation forest score density, and XAI feature importance rankings.
- **🔍 Real-Time Packet Inspector:** One-click presets for common attack vectors (SYN Flood, Portscan, Brute Force, Buffer Overflow, Zero-Day), granular network attribute adjustment, and live dual-engine inference.
- **🛡️ Actionable SOC Mitigation Playbooks:** Instant mitigation steps, MITRE ATT&CK technique IDs, and copy-paste Linux `iptables` & Snort IDS rules.
- **📁 Batch Network Log Profiler:** Bulk CSV/log file ingestion, dynamic severity filtering, full text search, and downloadable forensic CSV reports.
- **🧠 Model Diagnostics & Architecture:** Confusion matrix heatmap, precision/recall/F1 metrics per attack family, and interactive retraining triggers.

---

## 📂 Project Architecture

```
cybersecurity-threat-profiler/
├── dataset/
│   ├── download_dataset.py     # Script to download or generate NSL-KDD dataset
│   ├── sample_traffic.csv      # Sample batch traffic dataset for testing
│   └── KDDTrain+.txt           # NSL-KDD training file (auto-fetched or generated)
├── models/
│   ├── known_threat_model.pkl  # Serialized Supervised Threat Classifier
│   ├── zeroday_model.pkl       # Serialized Unsupervised Isolation Forest
│   ├── preprocessor.pkl        # Serialized Preprocessor (Encoders + Scaler)
│   └── metrics_summary.json    # Evaluation metrics & feature importances
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # NSL-KDD data loader & realistic synthetic generator
│   ├── preprocessor.py         # 42-feature schema parser, OHE & standard scaling
│   ├── train_models.py         # Model training pipeline & metric evaluation
│   └── threat_mitigation.py    # MITRE ATT&CK mapping, severity & firewall playbooks
├── app.py                      # Main Streamlit Dashboard application
├── test_pipeline.py            # Automated test suite
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open Workspace
Navigate to the project root directory:
```bash
cd cybersecurity-threat-profiler
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
Execute the end-to-end ML training pipeline to train and save the dual engines:
```bash
python src/train_models.py
```

### 4. Run the Streamlit Dashboard
Launch the web application:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the test suite to verify loading, preprocessing, model serialization, and inference:
```bash
python test_pipeline.py
```

---

## 🔬 NSL-KDD 5 Attack Classes Mapping

| Attack Family | Description | Common NSL-KDD Attacks Included | MITRE ATT&CK |
| :--- | :--- | :--- | :--- |
| **Normal** | Legitimate baseline traffic | `normal` | N/A |
| **DoS** | Denial of Service attacks | `neptune`, `smurf`, `back`, `teardrop`, `pod`, `land` | T1498 / T1499 |
| **Probe** | Surveillance & port scans | `ipsweep`, `portsweep`, `nmap`, `satan`, `mscan` | T1046 / T1595 |
| **R2L** | Remote to Local infiltration | `warezclient`, `guess_passwd`, `imap`, `ftp_write` | T1110 / T1190 |
| **U2R** | User to Root escalation | `buffer_overflow`, `rootkit`, `loadmodule`, `perl` | T1068 / T1548 |
| **Zero-Day** | Unseen statistical anomalies | Out-of-distribution behavioral drift | T1203 / T1059 |

---

## 📜 License
MIT License. Built for Cybersecurity Research & Network Defense Engineering.
