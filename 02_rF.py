import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics   import mean_absolute_error

df = pd.read_csv("campus_energy_data.csv")

X = df[["hour", "day_of_week", "is_weekend"]]
y = df["power_watts"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 42, test_size = 0.2
    )

model = LinearRegression()

model.fit(X_train, y_train)

predctions = model.predict(X_test)

compair_df = pd.DataFrame({
    "Predict" : predctions.round(2),
    "Actual" : y_test.values,
    })

compair_df["Difference"] = abs(compair_df["Actual"] - compair_df["Predict"]).round(2)

mae = mean_absolute_error(predctions, y_test)
print(f"MAE:  {mae:.2f}")

THRESHOLD = 1500
compair_df["IsAnatomy"] = compair_df["Difference"] > THRESHOLD
anotomies = compair_df[compair_df["IsAnatomy"] == True]

# print(compair_df.head().to_string(index = False))

print(f"Total: {len(compair_df)}")
print(f"Anotomies: {len(anotomies)}")










