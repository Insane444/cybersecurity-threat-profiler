"""
Threat Intelligence, MITRE ATT&CK Matrix, Multi-Vendor Rule Generator, and CISO Triage.
Provides rich contextual intelligence, multi-vendor SOC rules (iptables, nftables, Snort, Splunk, Cisco),
and automated forensic incident report generation.
"""

import time
import hashlib

# Comprehensive MITRE ATT&CK Matrix Stages
MITRE_TACTICS = [
    {"id": "TA0043", "name": "Reconnaissance", "active_in": ["Probe"]},
    {"id": "TA0001", "name": "Initial Access", "active_in": ["R2L", "Zero-Day"]},
    {"id": "TA0002", "name": "Execution", "active_in": ["U2R", "R2L", "Zero-Day"]},
    {"id": "TA0003", "name": "Persistence", "active_in": ["U2R"]},
    {"id": "TA0004", "name": "Privilege Escalation", "active_in": ["U2R"]},
    {"id": "TA0005", "name": "Defense Evasion", "active_in": ["Zero-Day", "Probe"]},
    {"id": "TA0006", "name": "Credential Access", "active_in": ["R2L"]},
    {"id": "TA0007", "name": "Discovery", "active_in": ["Probe"]},
    {"id": "TA0008", "name": "Lateral Movement", "active_in": ["R2L", "U2R"]},
    {"id": "TA0040", "name": "Impact", "active_in": ["DoS"]}
]

MITRE_ATTACK_MAPPINGS = {
    "Normal": {
        "severity": "Secure",
        "severity_level": 0,
        "color": "#10b981",  # Emerald
        "badge_bg": "rgba(16, 185, 129, 0.15)",
        "badge_border": "#10b981",
        "icon": "🛡️",
        "title": "Normal / Legitimate Traffic",
        "mitre_id": "N/A",
        "mitre_name": "Standard Network Operation",
        "summary": "Packet attributes adhere strictly to baseline protocol RFC standards and legitimate enterprise operational profiles.",
        "risk_level": "None (Healthy)",
        "defcon_level": "DEFCON 5 (Normal Readiness)",
        "recommendations": [
            "Maintain continuous telemetry ingestion and NetFlow logging.",
            "Verify perimeter default-deny egress filtering policies.",
            "Periodically audit TLS cipher suites and TLS 1.3 enforcement."
        ],
        "rules": {
            "iptables": "# Normal Verified Traffic Telemetry\niptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT",
            "nftables": "table inet filter {\n    chain input {\n        ct state { established, related } accept\n    }\n}",
            "snort": "pass ip any any -> $HOME_NET any (msg:\"Normal Verified Enterprise Flow\"; sid:1000001; rev:1;)",
            "splunk": "index=firewall action=allowed | stats count by src_ip, dst_port | head 10",
            "cisco": "access-list 101 permit ip any any established"
        }
    },
    "DoS": {
        "severity": "Critical",
        "severity_level": 4,
        "color": "#ef4444",  # Crimson
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "#ef4444",
        "icon": "🚨",
        "title": "Denial of Service (DoS) Attack",
        "mitre_id": "T1498 / T1499",
        "mitre_name": "Network Denial of Service & Endpoint Resource Exhaustion",
        "summary": "Adversary is flooding connection state tables, TCP SYN queues, or available ingress bandwidth to cause service degradation.",
        "risk_level": "Critical (Immediate Threat)",
        "defcon_level": "DEFCON 1 (Maximum Combat Readiness)",
        "recommendations": [
            "Enable kernel-level SYN flood protection (sysctl -w net.ipv4.tcp_syncookies=1).",
            "Engage upstream ISP / Cloudflare / AWS Shield automated DDoS scrubbing.",
            "Rate-limit incoming TCP new connection rates on boundary load balancers.",
            "Enforce strict drop on malformed fragmented packets and non-routable source IPs."
        ],
        "rules": {
            "iptables": "# Mitigate TCP SYN Flood & Malformed Packets\niptables -N SYN_FLOOD\niptables -A INPUT -p tcp --syn -j SYN_FLOOD\niptables -A SYN_FLOOD -m limit --limit 10/s --limit-burst 20 -j RETURN\niptables -A SYN_FLOOD -j DROP\niptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP\niptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP",
            "nftables": "table inet filter {\n    chain input {\n        tcp flags syn tcp dport { 80, 443 } limit rate 50/second accept\n        tcp flags syn tcp dport { 80, 443 } drop\n    }\n}",
            "snort": "alert tcp any any -> $HOME_NET any (flags: S; threshold: type threshold, track by_src, count 100, seconds 3; msg:\"CRITICAL - TCP SYN FLOOD DETECTED\"; sid:2000001; rev:3;)",
            "splunk": "index=firewall flags=S OR flags=S0 | stats count as syn_count by src_ip | where syn_count > 500 | sort - syn_count",
            "cisco": "access-list 105 deny tcp any any syn-flood-rate 200\naccess-list 105 permit ip any any"
        }
    },
    "Probe": {
        "severity": "High",
        "severity_level": 3,
        "color": "#f59e0b",  # Amber
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "#f59e0b",
        "icon": "⚠️",
        "title": "Surveillance & Network Probing",
        "mitre_id": "T1046 / T1595",
        "mitre_name": "Network Service Discovery & Active Scanning",
        "summary": "Adversary is conducting systematic port sweeping, OS fingerprinting, or vulnerability probing across network ranges.",
        "risk_level": "High (Pre-Attack Reconnaissance)",
        "defcon_level": "DEFCON 2 (Armed Forces Ready)",
        "recommendations": [
            "Dynamically blacklist reconnaissance IPs using automated fail2ban or dynamic ACLs.",
            "Deploy low-interaction honey-ports / tarpits to delay automated scanners.",
            "Disable ICMP timestamp and echo responses across untrusted edge boundaries.",
            "Review firewall access lists to verify no unassigned high ports are listening."
        ],
        "rules": {
            "iptables": "# Auto-drop Port Scanners for 15 Minutes\niptables -N PORTSCAN\niptables -A INPUT -m recent --name portscan_list --rcheck --seconds 900 -j DROP\niptables -A INPUT -m recent --name portscan_list --remove\niptables -A INPUT -p tcp -m tcp --tcp-flags RST,ACK,FIN,SYN RST -m limit --limit 2/s --limit-burst 5 -j RETURN\niptables -A INPUT -p tcp -m tcp --tcp-flags RST,ACK,FIN,SYN RST -m recent --name portscan_list --set -j DROP",
            "nftables": "table inet filter {\n    set scanner_blacklist { type ipv4_addr; flags timeout; }\n    chain input {\n        ip saddr @scanner_blacklist drop\n        tcp flags & (syn|ack) == syn update @scanner_blacklist { ip saddr timeout 15m } limit rate over 10/second drop\n    }\n}",
            "snort": "alert tcp any any -> $HOME_NET any (flags: SF,12; msg:\"SCAN - Stealth Nmap / Portsweep Signature\"; sid:2000002; rev:2;)",
            "splunk": "index=firewall action=blocked | stats dc(dst_port) as scanned_ports by src_ip | where scanned_ports > 25 | sort - scanned_ports",
            "cisco": "ip access-list extended RECON_BLOCK\n deny tcp host <SCANNER_IP> any\n permit ip any any"
        }
    },
    "R2L": {
        "severity": "High",
        "severity_level": 3,
        "color": "#a855f7",  # Purple
        "badge_bg": "rgba(168, 85, 247, 0.15)",
        "badge_border": "#a855f7",
        "icon": "🔓",
        "title": "Remote to Local (R2L) Infiltration",
        "mitre_id": "T1110 / T1190",
        "mitre_name": "Brute Force & Exploit Public-Facing Application",
        "summary": "Remote unauthorized user attempting credential guessing, password spray, guest login exploitation, or public daemon buffer attacks.",
        "risk_level": "High (Unauthorized Infiltration)",
        "defcon_level": "DEFCON 2 (Armed Forces Ready)",
        "recommendations": [
            "Trigger automated account lockout and require Multi-Factor Authentication (MFA).",
            "Immediately terminate all active remote sessions originated from the culprit IP.",
            "Inspect daemon authentication logs (sshd, vsftpd, dovecot, rdp) for repeated failures.",
            "Restrict management plane interfaces (SSH, RDP, Web-Admin) strictly to bastion VPNs."
        ],
        "rules": {
            "iptables": "# Restrict Brute Force & Rate Limit Remote Login Ports\niptables -A INPUT -p tcp -m multiport --dports 21,22,110,143 -m state --state NEW -m recent --set --name BRUTE_LIST\niptables -A INPUT -p tcp -m multiport --dports 21,22,110,143 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 --name BRUTE_LIST -j DROP",
            "nftables": "table inet filter {\n    chain input {\n        tcp dport { 21, 22, 110, 143 } ct state new meter auth_meter { ip saddr limit rate over 5/minute } drop\n    }\n}",
            "snort": "alert tcp any any -> $HOME_NET 21,22 (content:\"failed login\"; nocase; threshold: type both, track by_src, count 5, seconds 30; msg:\"AUTH - Rapid Failed Password Attempts\"; sid:2000003; rev:2;)",
            "splunk": "index=auth action=failure | stats count as fail_count by user, src_ip | where fail_count > 5 | sort - fail_count",
            "cisco": "access-list 110 deny tcp host <ATTACKER_IP> any eq 22\naccess-list 110 deny tcp host <ATTACKER_IP> any eq 21\naccess-list 110 permit ip any any"
        }
    },
    "U2R": {
        "severity": "Critical",
        "severity_level": 4,
        "color": "#ec4899",  # Pink-Red
        "badge_bg": "rgba(236, 72, 153, 0.15)",
        "badge_border": "#ec4899",
        "icon": "⚡",
        "title": "User to Root (U2R) Privilege Escalation",
        "mitre_id": "T1068 / T1548",
        "mitre_name": "Exploitation for Privilege Escalation & Sudo Abuse",
        "summary": "Local authenticated entity exploiting kernel vulnerability, su binary, setuid misconfiguration, or buffer overflow to attain superuser privileges.",
        "risk_level": "Critical (Host Compromise Imminent)",
        "defcon_level": "DEFCON 1 (Maximum Combat Readiness)",
        "recommendations": [
            "Immediately isolate the affected host from corporate VLANs via NAC/EDR quarantine.",
            "Terminate rogue child processes and revoke compromised user session tokens.",
            "Execute volatile memory capture for live kernel hook and shellcode forensic analysis.",
            "Audit all setuid/setgid binaries and examine recent sudoers / PAM alterations."
        ],
        "rules": {
            "iptables": "# Endpoint Quarantine Rule - Block All Outbound Network Egress\niptables -P OUTPUT DROP\niptables -A OUTPUT -o lo -j ACCEPT\niptables -A OUTPUT -p udp --dport 53 -j ACCEPT # Allow DNS for triage\n# Kill rogue spawned root sessions:\npkill -KILL -u <unprivileged_user>",
            "nftables": "table inet filter {\n    chain output {\n        type filter hook output priority 0; policy drop;\n        oif \"lo\" accept\n        udp dport 53 accept\n    }\n}",
            "snort": "alert ip any any -> $HOME_NET any (content:\"/bin/sh\"; nocase; content:\"uid=0(root)\"; msg:\"EXPLOIT - Root Shell Spawned on Network Endpoint\"; sid:2000004; rev:2;)",
            "splunk": "index=os (process=\"sudo\" OR process=\"su\") action=failure OR action=escalated | table _time, host, user, command",
            "cisco": "ep-quarantine host <HOST_MAC_ADDRESS> vlan 999"
        }
    },
    "Zero-Day": {
        "severity": "Elevated Anomaly",
        "severity_level": 3,
        "color": "#06b6d4",  # Cyan
        "badge_bg": "rgba(6, 182, 212, 0.18)",
        "badge_border": "#06b6d4",
        "icon": "🌀",
        "title": "Zero-Day / Novel Behavioral Anomaly",
        "mitre_id": "T1203 / T1059",
        "mitre_name": "Exploitation for Client Execution & Novel Zero-Day Vector",
        "summary": "Packet telemetry exhibits severe out-of-distribution drift from established benign baselines, indicating a novel zero-day exploit, polymorphic malware, or evasion protocol.",
        "risk_level": "Elevated (Novel Threat Vector)",
        "defcon_level": "DEFCON 2 (Armed Forces Ready)",
        "recommendations": [
            "Divert anomalous session to high-interaction honeynet / sandbox for behavioral detonation.",
            "Trigger automated Full Packet Capture (PCAP) for deep payload reverse engineering.",
            "Quarantine communicating endpoint and compare behavioral diff against benign cluster.",
            "Extract flow statistical anomalies to synthesize new Suricata signatures."
        ],
        "rules": {
            "iptables": "# Automated Sandbox Redirect & Full Flow Capture\ntcpdump -i eth0 -nn -s 0 -c 2000 -w /var/log/soc/zeroday_capture_$(date +%s).pcap\n# Quarantine flow by connection tuple:\niptables -I FORWARD 1 -s <ANOMALOUS_SRC_IP> -j DROP",
            "nftables": "table inet filter {\n    chain forward {\n        ip saddr <ANOMALOUS_SRC_IP> log prefix \"[ZERO-DAY-FLAG] \" drop\n    }\n}",
            "snort": "alert ip any any -> $HOME_NET any (msg:\"ZERO-DAY ANOMALY - Statistical Outlier Beyond 95th Percentile Baseline\"; sid:2000005; rev:2;)",
            "splunk": "index=network_anomaly anomaly_score > 0.85 | stats count by src_ip, dest_ip, app | sort - count",
            "cisco": "access-list 120 deny ip host <ANOMALOUS_SRC_IP> any log"
        }
    }
}


def get_threat_profile(category: str, is_anomaly: bool = False, anomaly_score: float = 0.0) -> dict:
    """Returns comprehensive threat profile, MITRE metadata, and mitigation playbooks."""
    clean_cat = str(category).strip()

    if clean_cat == "Normal" and is_anomaly:
        profile = MITRE_ATTACK_MAPPINGS["Zero-Day"].copy()
        profile["display_title"] = "Zero-Day / Novel Intrusion Anomaly"
        profile["fused_status"] = "Zero-Day Anomaly"
        profile["is_zeroday"] = True
    else:
        profile = MITRE_ATTACK_MAPPINGS.get(clean_cat, MITRE_ATTACK_MAPPINGS["Normal"]).copy()
        profile["display_title"] = profile["title"]
        profile["fused_status"] = clean_cat
        profile["is_zeroday"] = False

    profile["anomaly_score_pct"] = round(float(anomaly_score) * 100, 2)
    profile["firewall_rule"] = profile["rules"].get("iptables", "")
    profile["snort_rule"] = profile["rules"].get("snort", "")
    return profile


def generate_ciso_report(packet_data: dict, threat_profile: dict, confidence: float) -> str:
    """Generates an executive CISO Forensic Incident Report in Markdown."""
    incident_id = f"INC-{int(time.time())}-{hashlib.md5(str(packet_data).encode()).hexdigest()[:6].upper()}"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC")

    report = f"""# 🛡️ EXECUTIVE INCIDENT TRIAGE REPORT — {incident_id}

**Classification:** CONFIDENTIAL // SECURITY OPERATIONS CENTER  
**Generated At:** {timestamp}  
**Threat Status:** **{threat_profile['fused_status']}** (Severity: **{threat_profile['severity'].upper()}**)  
**DEFCON Threat Posture:** `{threat_profile.get('defcon_level', 'DEFCON 3')}`  

---

### 1. Executive Summary
During real-time network stream telemetry profiling, an event matching the **{threat_profile['title']}** signature was captured. The Supervised Multi-Class Classifier assessed this event with a **{confidence:.1f}% confidence index**, while the Unsupervised Isolation Forest registered an **Anomaly Score of {threat_profile.get('anomaly_score_pct', 0)}%**.

- **Primary MITRE ATT&CK Mapping:** `{threat_profile['mitre_id']}` (*{threat_profile['mitre_name']}*)
- **Threat Vector Description:** {threat_profile['summary']}

---

### 2. Forensic Telemetry Attributes
| Attribute | Value | Context |
| :--- | :--- | :--- |
| **Protocol Type** | `{packet_data.get('protocol_type', 'tcp').upper()}` | Layer 4 Transport |
| **Network Service** | `{packet_data.get('service', 'http')}` | Target Service Daemon |
| **TCP Connection Flag** | `{packet_data.get('flag', 'SF')}` | TCP Handshake Status |
| **Flow Duration** | `{packet_data.get('duration', 0)}s` | Total Connection Lifespan |
| **Source Payload Bytes** | `{packet_data.get('src_bytes', 0):,} B` | Outbound Transfer Volume |
| **Destination Payload Bytes** | `{packet_data.get('dst_bytes', 0):,} B` | Inbound Transfer Volume |
| **Concurrent Connection Count** | `{packet_data.get('count', 0)}` | Past 2s Window Volume |
| **SYN Error Rate** | `{packet_data.get('serror_rate', 0):.2f}` | Failed Handshake Metric |

---

### 3. Immediate SOC Containment Actions
"""
    for i, rec in enumerate(threat_profile["recommendations"], 1):
        report += f"{i}. {rec}\n"

    report += f"""
---

### 4. Generated Multi-Vendor Firewall & IDS Rules
#### Linux iptables:
```bash
{threat_profile['rules'].get('iptables', '')}
```

#### Linux nftables:
```bash
{threat_profile['rules'].get('nftables', '')}
```

#### Snort / Suricata 3.0 Signature:
```snort
{threat_profile['rules'].get('snort', '')}
```

#### Splunk SIEM Search Query:
```spl
{threat_profile['rules'].get('splunk', '')}
```

---
*Report generated by Cybersecurity Network Threat & Intrusion Profiler AI Engine.*
"""
    return report
