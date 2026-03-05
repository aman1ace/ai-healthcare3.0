import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# =========================
# HEART DISEASE MODEL
# =========================

print("Loading heart dataset...")

heart = pd.read_csv("../dataset/heart.csv")

# Replace ? with NaN
heart.replace("?", pd.NA, inplace=True)

# Convert all columns to numeric
heart = heart.apply(pd.to_numeric)

# Drop missing values
heart.dropna(inplace=True)

# Features and target
X_heart = heart.drop("num", axis=1)
y_heart = heart["num"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42
)

# Train model
heart_model = RandomForestClassifier()
heart_model.fit(X_train, y_train)

# Save model
joblib.dump(heart_model, "models/heart_model.pkl")

print("Heart disease model trained and saved.")


# =========================
# DIABETES MODEL
# =========================

print("Loading diabetes dataset...")

diabetes = pd.read_csv("../dataset/diabetes.csv")

# Features and target
X_dia = diabetes.drop("Outcome", axis=1)
y_dia = diabetes["Outcome"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_dia, y_dia, test_size=0.2, random_state=42
)

# Train model
diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_train, y_train)

# Save model
joblib.dump(diabetes_model, "models/diabetes_model.pkl")

print("Diabetes model trained and saved.")

print("All models trained successfully!")
