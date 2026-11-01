import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("dataset.csv")

df = pd.DataFrame(data)

# plt.hist(df["voltage_v"])
# print(df.columns)
# print(df.nunique())

plt.bar(df["device_id"], df["energy_kwh"])
plt.show()
