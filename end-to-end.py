# ===============================
# Heart Disease Prediction Project
# End-to-End Data Science Pipeline
# ===============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ===============================
# Load Dataset
# ===============================

df = pd.read_csv("heart.csv")

# ===============================
# Explore Dataset
# ===============================

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ===============================
# Data Cleaning
# ===============================

df = df.drop_duplicates()

# ===============================
# Exploratory Data Analysis
# ===============================

# Target Distribution
plt.figure(figsize=(6,4))
df["target"].value_counts().plot(kind="bar")
plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# Age Distribution
plt.figure(figsize=(7,5))
plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
corr = df.corr()

plt.figure(figsize=(10,8))
plt.imshow(corr)
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")
plt.show()

# ===============================
# Feature Selection
# ===============================

X = df.drop("target", axis=1)
y = df["target"]

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ===============================
# Feature Scaling
# ===============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# Train Model
# ===============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================

y_pred = model.predict(X_test)

# ===============================
# Evaluation
# ===============================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion Matrix
# ===============================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.title("Confusion Matrix")
plt.show()

# ===============================
# Feature Importance
# ===============================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

plt.figure(figsize=(10,6))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xticks(rotation=90)

plt.title("Feature Importance")

plt.show()

# ===============================
# Predict New Patient
# ===============================

sample = X.iloc[[0]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction: Heart Disease Detected")
else:
    print("\nPrediction: No Heart Disease")

# ===============================
# Save Model
# ===============================

import joblib

joblib.dump(model, "heart_model.pkl")

print("\nModel Saved Successfully!")
