
import numpy as np
import pandas as pd

# Reproducible dataset
np.random.seed(42)

# Dataset configuration
total_records = 3000

timestamps = pd.date_range(
    start="2026-01-01",
    periods=total_records,
    freq="15min"
)

# Generate lab usage conditions
hours = timestamps.hour

occupancy = []

for hour in hours:
    if 9 <= hour <= 16:
        occupancy.append(np.random.randint(10, 61))
    else:
        occupancy.append(np.random.randint(0, 6))

occupancy = np.array(occupancy)

active_computers = np.minimum(
    occupancy,
    np.random.randint(1, 61, total_records)
)

# Electrical measurements
voltage = np.random.normal(230, 4, total_records)

current = (
    0.15
    + active_computers * 0.12
    + np.random.normal(0, 0.3, total_records)
)

power_factor = np.random.normal(0.92, 0.04, total_records)
power_factor = np.clip(power_factor, 0.65, 0.99)

power = voltage * current * power_factor

temperature = np.random.normal(28, 3, total_records)

frequency = np.random.normal(50, 0.1, total_records)

# Energy consumed during 15 minutes
energy = power * 0.25 / 1000

# Initially, all records are normal
anomaly_label = np.zeros(total_records, dtype=int)

# Create unusual electricity consumption
anomaly_indices = np.random.choice(
    total_records, 90, replace=False
)

power[anomaly_indices] *= np.random.uniform(
    1.8, 3.0, len(anomaly_indices)
)

current[anomaly_indices] *= np.random.uniform(
    1.8, 3.0, len(anomaly_indices)
)

anomaly_label[anomaly_indices] = 1

# Create dataset
df = pd.DataFrame({
    "timestamp": timestamps,
    "hour": hours,
    "occupancy_count": occupancy,
    "active_computers": active_computers,
    "voltage_v": voltage,
    "current_a": current,
    "active_power_w": power,
    "energy_kwh": energy,
    "power_factor": power_factor,
    "frequency_hz": frequency,
    "temperature_c": temperature,
    "anomaly_label": anomaly_label
})

# Introduce duplicate records
duplicates = df.sample(25, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# Introduce missing values
for column in ["voltage_v", "current_a", "temperature_c",
               "power_factor"]:
    indices = np.random.choice(df.index, 35, replace=False)
    df.loc[indices, column] = np.nan

# Introduce incorrect and extreme values
df.loc[10, "voltage_v"] = -230
df.loc[20, "current_a"] = 500
df.loc[30, "power_factor"] = 1.8
df.loc[40, "temperature_c"] = -100

# Introduce incorrect data type
df["active_power_w"] = df["active_power_w"].astype(object)
df.loc[50, "active_power_w"] = "2500W"

# Shuffle records
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save raw dataset
df.to_csv("dataset/campus_energy_raw.csv", index=False)

print("Raw dataset created successfully!")
print("Shape:", df.shape)
print("\nFirst five records:")
print(df.head())
