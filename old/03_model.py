import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("dataset.csv")

df = pd.DataFrame(data)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(df.describe())

