import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 100_000

age = rng.integers(21, 66, N)
income = np.clip(rng.lognormal(10.9, 0.5, N), 120000, 5000000).round(0)
loan_amount = np.clip(income * rng.uniform(0.3, 3.0, N), 50000, 8000000).round(0)
loan_term_months = rng.choice([12, 24, 36, 48, 60, 84, 120], N)
credit_score = np.clip(rng.normal(690, 70, N), 300, 900).round(0)
employment_years = np.clip(age - 21 - rng.integers(0, 8, N), 0, 40)
num_existing_loans = rng.poisson(1.2, N)
num_late_payments = rng.poisson(np.clip((750 - credit_score) / 120, 0.05, None))
credit_utilization = np.clip(rng.beta(2, 3, N), 0, 1).round(3)
dependents = rng.integers(0, 6, N)
education = rng.choice(["High School", "Graduate", "Postgraduate"], N, p=[0.3, 0.5, 0.2])
employment_type = rng.choice(["Salaried", "Self-Employed", "Business", "Unemployed"], N, p=[0.55, 0.2, 0.2, 0.05])
home_ownership = rng.choice(["Rent", "Own", "Mortgage"], N, p=[0.4, 0.3, 0.3])
loan_purpose = rng.choice(["Home", "Car", "Education", "Personal", "Business"], N)
marital_status = rng.choice(["Single", "Married", "Divorced"], N, p=[0.35, 0.55, 0.1])

df = pd.DataFrame({
    "age": age, "income": income, "loan_amount": loan_amount,
    "loan_term_months": loan_term_months, "credit_score": credit_score,
    "employment_years": employment_years, "num_existing_loans": num_existing_loans,
    "num_late_payments": num_late_payments, "credit_utilization": credit_utilization,
    "dependents": dependents, "education": education,
    "employment_type": employment_type, "home_ownership": home_ownership,
    "loan_purpose": loan_purpose, "marital_status": marital_status,
})

# Default probability from a hidden logistic rule + noise (so the ANN has real signal to learn)
dti = loan_amount / income
z = (-2.2
     + 2.8 * (dti - 1.2)
     - 0.012 * (credit_score - 690)
     + 0.35 * num_late_payments
     + 1.5 * (credit_utilization - 0.4)
     + 0.15 * num_existing_loans
     - 0.03 * employment_years
     + 1.2 * (employment_type == "Unemployed")
     + 0.2 * (employment_type == "Self-Employed")
     - 0.2 * (home_ownership == "Own")
     + 0.1 * dependents
     + rng.normal(0, 0.6, N))
p = 1 / (1 + np.exp(-z))
df["default"] = (rng.random(N) < p).astype(int)

# Add a few missing values to practice preprocessing
for col in ["income", "employment_years", "credit_score"]:
    df.loc[rng.choice(N, int(N * 0.01), replace=False), col] = np.nan

df.to_csv("loan_default_100k.csv", index=False)
print(df.shape, "default rate:", df["default"].mean().round(3))
