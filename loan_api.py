# loan_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------
# Globals
# -------------------
PIPELINE_PATH = "loan_pipeline.pkl"
DATA_PATH = "train.csv"  # replace with your dataset path
pipe = None

# Expected JSON schema (keys and example values)
EXPECTED_FIELDS = {
    "Gender": "Male",                 
    "Married": "Yes",                 
    "Dependents": 0,                  
    "Education": "Graduate",          
    "Self_Employed": "No",            
    "ApplicantIncome": 5849,          
    "CoapplicantIncome": 0,           
    "LoanAmount": 128,                
    "Loan_Amount_Term": 360,          
    "Credit_History": 1,              
    "Property_Area": "Urban"          
}

# -------------------
# Init model function
# -------------------
def init_model():
    global pipe
    if os.path.exists(PIPELINE_PATH):
        print("Loading existing pipeline...")
        pipe = joblib.load(PIPELINE_PATH)
        return pipe
    
    print("Training new pipeline...")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Drop Loan_ID if exists
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Fill missing values
    for col in ["Gender","Married","Dependents","Self_Employed","Credit_History"]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in ["LoanAmount","Loan_Amount_Term","ApplicantIncome","CoapplicantIncome"]:
        df[col].fillna(df[col].median(), inplace=True)

    # Encode target
    df["Loan_Status"] = df["Loan_Status"].map({"Y":1, "N":0})

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Categorical & numerical cols
    cat_cols = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
    num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]

    # Transformers
    cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                      ("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    # Final pipeline
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", LogisticRegression(max_iter=200))])

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    # Save pipeline
    joblib.dump(pipe, PIPELINE_PATH)
    print("Pipeline trained and saved.")

    return pipe

# -------------------
# Flask App
# -------------------
app = Flask(__name__)
CORS(app)  

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": PIPELINE_PATH}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Minimal validation
        missing = [k for k in EXPECTED_FIELDS.keys() if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}", "expected": EXPECTED_FIELDS}), 400

        # Handle "3+"
        dep = data.get("Dependents")
        if isinstance(dep, str) and dep.strip() == "3+":
            data["Dependents"] = 3

        # Ensure numeric
        numeric_fields = ["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]
        for nf in numeric_fields:
            try:
                data[nf] = float(data[nf]) if data[nf] is not None else None
            except Exception:
                return jsonify({"error": f"Field '{nf}' must be numeric."}), 400

        # Create DataFrame
        row = pd.DataFrame([data])

        # Predict
        pred = pipe.predict(row)[0]
        proba = None
        if hasattr(pipe, "predict_proba"):
            proba_arr = pipe.predict_proba(row)[0]
            proba = float(proba_arr[1])

        result = {
            "loan_status": "Approved" if int(pred) == 1 else "Rejected",
            "approval_probability": proba
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    pipe = init_model()   # Initialize or train model at startup
    app.run(host="0.0.0.0", port=5000, debug=True)
