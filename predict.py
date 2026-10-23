import joblib
import pandas as pd
from tensorflow import keras

model = keras.models.load_model("loan_ann_model.keras")
pre = joblib.load("preprocessor.joblib")

new_applicant = pd.DataFrame([{
    "age": 35, "income": 600000, "loan_amount": 900000,
    "loan_term_months": 60, "credit_score": 710,
    "employment_years": 10, "num_existing_loans": 1,
    "num_late_payments": 0, "credit_utilization": 0.3,
    "dependents": 2, "education": "Graduate",
    "employment_type": "Salaried", "home_ownership": "Rent",
    "loan_purpose": "Car", "marital_status": "Married",
}])

x = pre.transform(new_applicant)
if hasattr(x, "toarray"):
    x = x.toarray()

prob = float(model.predict(x)[0][0])
print(f"Default probability: {prob:.2%}")
print("Decision:", "REJECT (high risk)" if prob > 0.5 else "APPROVE (low risk)")
