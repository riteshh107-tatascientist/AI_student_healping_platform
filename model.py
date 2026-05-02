import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"D:\AI_prediction_Hackthon\Student_Performance.csv")

# Drop ID column
df = df.drop("student_id", axis=1)

# Categorical columns
cat_cols = [
    "gender",
    "school_type",
    "parent_education",
    "internet_access",
    "travel_time",
    "extra_activities",
    "study_method"
]

encoders = {}

# Encode categorical data
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Encode target
target_le = LabelEncoder()
df["final_grade"] = target_le.fit_transform(df["final_grade"])

# Features & target
X = df.drop("final_grade", axis=1)
y = df["final_grade"]

# 🚀 IMPORTANT FIX (ADD THIS)
feature_columns = X.columns
joblib.dump(feature_columns, "features.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_le, "target_encoder.pkl")

print("✅ Model trained successfully!")
print("📦 model.pkl, encoders.pkl, target_encoder.pkl, features.pkl saved")