"""
Live Traffic Stream Simulator and Packet Dissection Engine.
Simulates real-time network packet streams, PCAP header dissection,
and raw hex/ASCII payload representation for security analysis.
"""

import time
import random
import numpy as np
import pandas as pd
from src.preprocessor import PROTOCOLS, TOP_SERVICES, TOP_FLAGS, map_attack_type

# Simulated realistic IP pools
SRC_IPS = [
    "192.168.1.45", "10.0.0.12", "172.16.4.88", "192.168.1.105", 
    "45.33.32.156", "185.220.101.5", "91.240.118.22", "198.51.100.23",
    "203.0.113.195", "103.235.46.19", "185.191.171.12", "194.26.29.112"
]
DST_IPS = [
    "10.0.0.1", "10.0.0.50", "192.168.1.1", "192.168.1.254",
    "172.16.0.10", "172.16.0.25", "10.0.1.100", "10.0.2.200"
]

SERVICE_PORT_MAP = {
    "http": 80, "smtp": 25, "ftp": 21, "ftp_data": 20, "domain_u": 53,
    "telnet": 23, "finger": 79, "eco_i": 8, "ecr_i": 0, "auth": 113,
    "private": 445, "pop_3": 110, "ssh": 22, "dns": 53, "ntp": 123,
    "sunrpc": 111, "time": 37, "other": 8080
}


def generate_simulated_packet(force_attack: str = None) -> dict:
    """
    Generates a realistic single network packet record with both
    NSL-KDD attributes and live network telemetry (IPs, Ports, TTL, TCP Flags).
    """
    attack_types = ["Normal", "Normal", "Normal", "Normal", "DoS", "Probe", "R2L", "U2R", "Zero-Day"]
    chosen_class = force_attack or random.choice(attack_types)

    src_ip = random.choice(SRC_IPS)
    dst_ip = random.choice(DST_IPS)
    timestamp = time.strftime("%H:%M:%S")

    if chosen_class == "Normal":
        proto = random.choice(["tcp", "tcp", "tcp", "udp", "icmp"])
        srv = random.choice(["http", "http", "smtp", "domain_u", "ftp_data"])
        flag = "SF"
        src_bytes = int(np.random.exponential(500) + 120)
        dst_bytes = int(np.random.exponential(2200) + 200)
        count = random.randint(1, 10)
        srv_count = random.randint(1, 10)
        serror = 0.0
        rerror = 0.0
        diff_srv = float(np.random.uniform(0.0, 0.1))
        same_srv = float(np.random.uniform(0.9, 1.0))
        logged_in = 1 if proto == "tcp" and srv in ["http", "smtp"] else 0
        raw_attack = "normal"

    elif chosen_class == "DoS":
        proto = random.choice(["tcp", "icmp"])
        srv = random.choice(["private", "http", "eco_i", "ecr_i"])
        flag = random.choice(["S0", "REJ", "SF"])
        src_bytes = random.choice([0, 0, 44, 1032])
        dst_bytes = 0
        count = random.randint(180, 510)
        srv_count = random.randint(10, 510)
        serror = 1.0 if flag == "S0" else 0.0
        rerror = 1.0 if flag == "REJ" else 0.0
        diff_srv = 0.05
        same_srv = 0.95
        logged_in = 0
        raw_attack = random.choice(["neptune", "smurf", "teardrop"])

    elif chosen_class == "Probe":
        proto = random.choice(["tcp", "icmp", "udp"])
        srv = random.choice(["private", "telnet", "finger", "ftp"])
        flag = random.choice(["REJ", "RSTO", "S0"])
        src_bytes = random.choice([0, 8, 44])
        dst_bytes = random.choice([0, 40])
        count = random.randint(10, 80)
        srv_count = random.randint(1, 5)
        serror = 0.0
        rerror = float(np.random.uniform(0.5, 1.0))
        diff_srv = float(np.random.uniform(0.7, 1.0))
        same_srv = float(np.random.uniform(0.0, 0.2))
        logged_in = 0
        raw_attack = random.choice(["ipsweep", "portsweep", "nmap", "satan"])

    elif chosen_class == "R2L":
        proto = "tcp"
        srv = random.choice(["ftp", "telnet", "smtp"])
        flag = "SF"
        src_bytes = random.randint(300, 2500)
        dst_bytes = random.randint(600, 6000)
        count = random.randint(1, 6)
        srv_count = random.randint(1, 6)
        serror = 0.0
        rerror = 0.0
        diff_srv = 0.0
        same_srv = 1.0
        logged_in = random.choice([0, 1])
        raw_attack = random.choice(["guess_passwd", "warezclient", "ftp_write"])

    elif chosen_class == "U2R":
        proto = "tcp"
        srv = random.choice(["telnet", "ftp_data"])
        flag = "SF"
        src_bytes = random.randint(600, 3500)
        dst_bytes = random.randint(1500, 12000)
        count = random.randint(1, 4)
        srv_count = random.randint(1, 4)
        serror = 0.0
        rerror = 0.0
        diff_srv = 0.0
        same_srv = 1.0
        logged_in = 1
        raw_attack = random.choice(["buffer_overflow", "rootkit"])

    else:  # Zero-Day
        proto = random.choice(["udp", "tcp"])
        srv = random.choice(["other", "private"])
        flag = random.choice(["OTH", "SHR"])
        src_bytes = random.randint(12000, 65000)
        dst_bytes = random.randint(5, 50)
        count = random.randint(30, 90)
        srv_count = random.randint(1, 5)
        serror = 0.0
        rerror = 0.0
        diff_srv = 0.95
        same_srv = 0.05
        logged_in = 0
        raw_attack = "unknown_polymorphic_zero_day"

    src_port = random.randint(1024, 65535)
    dst_port = SERVICE_PORT_MAP.get(srv, 80)

    packet = {
        "timestamp": timestamp,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "duration": random.randint(0, 5) if chosen_class == "Normal" else random.randint(0, 45),
        "protocol_type": proto,
        "service": srv,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": random.randint(1, 5) if chosen_class in ["R2L", "U2R"] else 0,
        "num_failed_logins": random.randint(1, 4) if chosen_class == "R2L" else 0,
        "logged_in": logged_in,
        "num_compromised": random.randint(1, 5) if chosen_class == "U2R" else 0,
        "root_shell": 1 if chosen_class == "U2R" else 0,
        "su_attempted": 1 if chosen_class == "U2R" else 0,
        "num_root": random.randint(1, 8) if chosen_class == "U2R" else 0,
        "num_file_creations": random.randint(1, 3) if chosen_class == "U2R" else 0,
        "num_shells": 1 if chosen_class == "U2R" else 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 1 if chosen_class == "R2L" and random.random() < 0.5 else 0,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": serror,
        "srv_serror_rate": serror,
        "rerror_rate": rerror,
        "srv_rerror_rate": rerror,
        "same_srv_rate": same_srv,
        "diff_srv_rate": diff_srv,
        "srv_diff_host_rate": float(np.random.uniform(0.0, 0.4)),
        "dst_host_count": 255 if chosen_class in ["DoS", "Probe"] else random.randint(20, 150),
        "dst_host_srv_count": 20 if chosen_class in ["DoS", "Probe"] else random.randint(50, 255),
        "dst_host_same_srv_rate": same_srv,
        "dst_host_diff_srv_rate": diff_srv,
        "dst_host_same_src_port_rate": 0.85 if chosen_class == "Probe" else 0.05,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": serror,
        "dst_host_srv_serror_rate": serror,
        "dst_host_rerror_rate": rerror,
        "dst_host_srv_rerror_rate": rerror,
        "attack_type": raw_attack,
        "level": 21
    }
    return packet


def generate_packet_hex_dump(packet: dict) -> str:
    """Generates an authentic raw Hex / ASCII packet dissection."""
    proto = packet.get("protocol_type", "tcp").upper()
    src_ip = packet.get("src_ip", "192.168.1.45")
    dst_ip = packet.get("dst_ip", "10.0.0.1")
    src_port = packet.get("src_port", 44321)
    dst_port = packet.get("dst_port", 80)
    flag = packet.get("flag", "SF")
    src_bytes = packet.get("src_bytes", 350)

    hex_lines = [
        f"0000   00 50 56 c0 00 08 00 0c 29 3e 5b 7d 08 00 45 00   .PV...)>[}}.E.",
        f"0010   00 3c 7a 1b 40 00 40 06 {random.randint(10, 99)} 8b c0 a8 01 2d 0a 00   .<z.@.@.......-..",
        f"0020   00 01 {src_port:04x} {dst_port:04x} 1b 4e 82 f0 00 00 00 00 a0 02   ....N.........",
        f"0030   72 10 9a 3d 00 00 02 04 05 b4 04 02 08 0a {random.randint(10, 99):02x} {random.randint(10, 99):02x}   r..=..........",
        f"0040   {random.randint(10, 99):02x} {random.randint(10, 99):02x} 00 00 00 00 01 03 03 07 47 45 54 20 2f 20   ..........GET / ",
        f"0050   48 54 54 50 2f 31 2e 31 0d 0a 48 6f 73 74 3a 20   HTTP/1.1..Host: ",
        f"0060   31 30 2e 30 2e 30 2e 31 0d 0a 55 73 65 72 2d 41   10.0.0.1..User-A"
    ]
    return "\n".join(hex_lines)
