# train_pipeline.py
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---- Load data ----
df = pd.read_csv("train.csv")

# Basic cleaning
df = df.copy()
# Dependents: '3+' -> 3 (int)
df["Dependents"] = df["Dependents"].replace("3+", 3)
# Convert numeric-like columns to numeric
num_like = ["Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
for c in num_like:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Target & features
y = df["Loan_Status"].map({"Y": 1, "N": 0})   # binary target
X = df[[
    "Gender","Married","Dependents","Education","Self_Employed",
    "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",
    "Credit_History","Property_Area"
]]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Columns by type
numeric_features = ["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]
categorical_features = ["Gender","Married","Education","Self_Employed","Property_Area"]

# Preprocess
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Model
clf = RandomForestClassifier(random_state=42)

# Full pipeline
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", clf)
])

# Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the pipeline (preprocessing + model together)
joblib.dump(pipe, "loan_pipeline.pkl")
print("Saved: loan_pipeline.pkl")
