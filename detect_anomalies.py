import numpy as np
import pandas as pd

# ---------------------------------------
# CONFIG
# ---------------------------------------
ROLLING_WINDOW = 7
MIN_ABS_INCREASE = 500

# ---------------------------------------
# ANOMALY DETECTION
# ---------------------------------------
def detect_anomalies(df):
    values = df["Cases"].values
    mean, std = values.mean(), values.std()

    spike_idx = [
        i for i, v in enumerate(values)
        if abs(v - mean) > 3 * std
    ]

    growth = np.diff(values) / np.maximum(values[:-1], 1)
    growth_idx = [i + 1 for i, g in enumerate(growth) if g > 0.4]

    anomalies = set(spike_idx + growth_idx)
    df["Anomaly"] = ["YES" if i in anomalies else "NO" for i in range(len(df))]

    return df


# ---------------------------------------
# SEVERITY CLASSIFICATION
# ---------------------------------------
def compute_severity(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["Severity"] = ""
    df["Agent Decision"] = ""
    df["Action"] = ""
    for i in range(len(df)):
        if df.loc[i, "Anomaly"] == "YES":
            if i < ROLLING_WINDOW:
                df.loc[i, "Severity"] = ""

            curr = df.loc[i, "Cases"]
            baseline = df.loc[i - ROLLING_WINDOW:i- 1, "Cases"].mean()

            abs_inc = curr - baseline
            growth = abs_inc / max(baseline, 1)

            if abs_inc < MIN_ABS_INCREASE:
                df.loc[i, "Severity"] = ""
            if growth >= 1.0:
                df.loc[i, "Severity"] = "CRITICAL"
            elif growth >= 0.4:
                df.loc[i, "Severity"] = "WARNING"
            else:
                df.loc[i, "Severity"] = "MINOR"
    return df

# ---------------------------------------
# OBSERVATION BUILDER
# ---------------------------------------
def build_observation(df, idx):
    return {
        "date": str(df.loc[idx, "Date"].date()),
        "cases": int(df.loc[idx, "Cases"]),
        "severity": df.loc[idx, "Severity"]
    }

# ---------------------------------------
# Agent ACTION DECIDER
# ---------------------------------------
def agent_action(df, idx,action):
    df.loc[idx, "Agent Decision"] = action

    if action == "FIX_ANOMALY":
        fix_anomaly(df, idx)

    elif action == "KEEP_ANOMALY":
        df.loc[idx, "Action"] = "Accepted as a real outbreak signal"

    elif action == "FLAG_FOR_REVIEW":
        df.loc[idx, "Action"] = "Flagged for human review"
    return df

# ---------------------------------------
# FIX ANOMALY
# ---------------------------------------

def fix_anomaly(df, idx):
    window = df.loc[max(0, idx - 3):idx - 1, "Cases"]
    if len(window) > 0:
        df.loc[idx, "Cases"] = int(window.mean())

    df.loc[idx, "Severity"] = ""
    df.loc[idx, "Action"] = "Auto-corrected by an AI agent"


