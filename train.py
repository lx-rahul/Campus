import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("dataset/campus_energy_cleaned.csv")


df = df.sort_values("timestamp").reset_index(drop=True)


X = df.drop(columns=["anomaly_label", "timestamp"])
y = df["anomaly_label"]


split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


model.fit(X_train, y_train)


# ==================================
# 7. MAKE PROBABILITY-BASED PREDICTIONS
# ==================================

y_prob = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.5

y_pred = (y_prob >= THRESHOLD).astype(int)

print("\nThreshold:", THRESHOLD)

print("\nFirst 20 anomaly probabilities:")
print(y_prob[:20])

print("\nFirst 20 predictions:")
print(y_pred[:20])


# ==================================
# 8. EVALUATE THE MODEL
# ==================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# ==================================
# 9. CONFUSION MATRIX
# ==================================

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)


# ==================================
# 10. CLASSIFICATION REPORT
# ==================================

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    zero_division=0
))


# ==================================
# 11. CHECK FEATURE IMPORTANCE
# ==================================

feature_imp = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_imp.round(4))


# ==================================
# 12. SAVE TRAINED MODEL
# ==================================

MODEL_PATH = Path("models/campuspulse_rf_model.joblib")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Model location:", MODEL_PATH)
