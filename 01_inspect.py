
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==========================================
# 1. LOAD DATASET
# ==========================================

RAW_PATH = Path("dataset/campus_energy_raw.csv")
CLEAN_PATH = Path("dataset/campus_energy_cleaned.csv")

df = pd.read_csv(RAW_PATH)

print("Original dataset:", df.shape)


# ==========================================
# 2. TYPE CONVERSION
# ==========================================

# Convert timestamp
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

# Clean active_power_w unit strings
df["active_power_w"] = (
    df["active_power_w"]
    .astype("string")
    .str.replace(
        r"(?i)\s*(watts|watt|w)\s*$",
        "",
        regex=True
    )
    .str.strip()
)

# Convert numerical columns
NUMERIC_COLS = [
    "hour",
    "occupancy_count",
    "active_computers",
    "voltage_v",
    "current_a",
    "active_power_w",
    "energy_kwh",
    "power_factor",
    "frequency_hz",
    "temperature_c",
    "anomaly_label"
]

for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ==========================================
# 3. HANDLE INVALID VALUES
# ==========================================

# Physical sensor limits
VALID_RANGES = {
    "voltage_v": (150, 280),
    "current_a": (0, 100),
    "active_power_w": (0, 25000),
    "energy_kwh": (0, None),
    "power_factor": (0, 1),
    "frequency_hz": (47, 53),
    "temperature_c": (0, 60),
    "occupancy_count": (0, 70),
    "active_computers": (0, 70)
}

for col, (low, high) in VALID_RANGES.items():

    if low is not None:
        df.loc[df[col] < low, col] = np.nan

    if high is not None:
        df.loc[df[col] > high, col] = np.nan


# Validate hour
df["hour"] = df["timestamp"].dt.hour

# Validate anomaly label
df.loc[
    ~df["anomaly_label"].isin([0, 1]),
    "anomaly_label"
] = np.nan


# ==========================================
# 4. HANDLE TIMESTAMP
# ==========================================

# Remove rows with invalid timestamps
df = df.dropna(subset=["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp")


# ==========================================
# 5. REMOVE DUPLICATES
# ==========================================

# Remove only completely identical rows
df = df.drop_duplicates()

# Do not remove rows just because timestamps match.
# Multiple readings may share a timestamp.


# ==========================================
# 6. HANDLE MISSING SENSOR VALUES
# ==========================================

SENSOR_COLS = [
    "voltage_v",
    "current_a",
    "active_power_w",
    "energy_kwh",
    "power_factor",
    "frequency_hz",
    "temperature_c"
]

# Fill short gaps using previous readings
df[SENSOR_COLS] = df[SENSOR_COLS].ffill(limit=3)

# Remove rows where important values are still missing
REQUIRED_COLS = [
    "timestamp",
    "hour",
    "occupancy_count",
    "active_computers",
    "anomaly_label"
] + SENSOR_COLS

df = df.dropna(subset=REQUIRED_COLS)

df = df.reset_index(drop=True)


# ==========================================
# 7. RESTORE INTEGER DATA TYPES
# ==========================================

INT_COLS = [
    "hour",
    "occupancy_count",
    "active_computers",
    "anomaly_label"
]

for col in INT_COLS:
    df[col] = df[col].astype(int)


# ==========================================
# 8. OUTLIER DETECTION USING IQR
# ==========================================

NUMERIC_FEATURES = [
    "occupancy_count",
    "active_computers",
    "voltage_v",
    "current_a",
    "active_power_w",
    "energy_kwh",
    "power_factor",
    "frequency_hz",
    "temperature_c"
]

print("\nOUTLIER DETECTION")

outlier_summary = {}

for col in NUMERIC_FEATURES:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    outlier_summary[col] = len(outliers)

    print(f"\n{col}")
    print("Lower limit:", lower)
    print("Upper limit:", upper)
    print("Potential outliers:", len(outliers))


print("\nOUTLIER SUMMARY")
print(pd.Series(outlier_summary))


# ==========================================
# 9. VISUALIZE OUTLIERS
# ==========================================

# for col in NUMERIC_FEATURES:
#
#     plt.figure(figsize=(8, 4))
#
#     sns.boxplot(x=df[col])
#
#     plt.title(f"{col} - Outlier Detection")
#     plt.xlabel(col)
#
#     plt.tight_layout()
#     plt.show()


# ==========================================
# 10. LOGICAL CONSISTENCY CHECKS
# ==========================================

print("\nLOGICAL CONSISTENCY CHECKS")

print(
    "Active computers greater than 70:",
    (df["active_computers"] > 70).sum()
)

print(
    "Occupancy greater than 70:",
    (df["occupancy_count"] > 70).sum()
)

print(
    "Active computers greater than occupancy:",
    (df["active_computers"] > df["occupancy_count"]).sum()
)

print(
    "Duplicate timestamps:",
    df["timestamp"].duplicated().sum()
)


# ==========================================
# 11. FINAL VERIFICATION
# ==========================================

print("\nFINAL DATASET SHAPE:")
print(df.shape)

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nSTATISTICAL SUMMARY:")
print(df.describe().T)


# ==========================================
# 12. SAVE CLEANED DATASET
# ==========================================

CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(CLEAN_PATH, index=False)

print("\nPreprocessing completed!")
print("Cleaned dataset saved at:", CLEAN_PATH)
