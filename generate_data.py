"""
=============================================
SYNTHETIC LOAN DATASET GENERATOR
=============================================
Generates a realistic loan application dataset
for the Loan Approval Predictor project.

Author: Vijayaragavan U
=============================================
"""

import pandas as pd
import numpy as np

np.random.seed(42)

NUM_SAMPLES = 614

loan_ids = [f'LP{str(i).zfill(4)}' for i in range(1001, 1001 + NUM_SAMPLES)]

gender = np.random.choice(['Male', 'Female'], size=NUM_SAMPLES, p=[0.80, 0.20])

married = np.where(
    gender == 'Male',
    np.random.choice(['Yes', 'No'], size=NUM_SAMPLES, p=[0.70, 0.30]),
    np.random.choice(['Yes', 'No'], size=NUM_SAMPLES, p=[0.50, 0.50])
)

dependents = np.random.choice(['0', '1', '2', '3+'], size=NUM_SAMPLES, p=[0.40, 0.25, 0.20, 0.15])

education = np.random.choice(['Graduate', 'Not Graduate'], size=NUM_SAMPLES, p=[0.50, 0.50])

self_employed = np.random.choice(['Yes', 'No'], size=NUM_SAMPLES, p=[0.15, 0.85])

applicant_income = np.random.lognormal(mean=8.5, sigma=0.6, size=NUM_SAMPLES).astype(int)
applicant_income = np.clip(applicant_income, 150, 100000)

coapplicant_income = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    if np.random.random() < 0.35:
        coapplicant_income[i] = 0
    else:
        coapplicant_income[i] = max(0, int(np.random.lognormal(mean=7.8, sigma=0.5)))

loan_amount = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    base = np.random.lognormal(mean=5.0, sigma=0.5)
    income_factor = 1.0
    if applicant_income[i] > 10000:
        income_factor = 1.3
    elif applicant_income[i] > 5000:
        income_factor = 1.1
    loan_amount[i] = max(9, min(700, int(base * income_factor)))

loan_amount = loan_amount.astype(int)

loan_amount_term = np.random.choice([360, 180, 120, 60, 84, 240, 300], size=NUM_SAMPLES,
                                     p=[0.45, 0.25, 0.12, 0.06, 0.05, 0.04, 0.03])

credit_history = np.random.choice([1.0, 0.0], size=NUM_SAMPLES, p=[0.75, 0.25])

property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], size=NUM_SAMPLES, p=[0.35, 0.35, 0.30])

loan_status = []
for i in range(NUM_SAMPLES):
    score = 0.0

    if credit_history[i] == 1.0:
        score += 0.25
    else:
        score -= 0.35

    income = applicant_income[i]
    if income > 10000:
        score += 0.15
    elif income > 5000:
        score += 0.08
    elif income < 2000:
        score -= 0.10

    co_income = coapplicant_income[i]
    if co_income > 5000:
        score += 0.08
    elif co_income > 0:
        score += 0.03

    amount = loan_amount[i]
    if amount < 100:
        score += 0.08
    elif amount > 300:
        score -= 0.10

    if property_area[i] == 'Semiurban':
        score += 0.08
    elif property_area[i] == 'Urban':
        score += 0.03

    if education[i] == 'Graduate':
        score += 0.05

    if self_employed[i] == 'Yes':
        score += 0.02

    if married[i] == 'Yes':
        score += 0.03

    base_prob = 0.42 + score + np.random.uniform(-0.10, 0.10)
    base_prob = np.clip(base_prob, 0.05, 0.92)

    loan_status.append('Y' if np.random.random() < base_prob else 'N')

loan_status = np.array(loan_status)

df = pd.DataFrame({
    'Loan_ID': loan_ids,
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income.astype(int),
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_area,
    'Loan_Status': loan_status
})

df.to_csv('data/train.csv', index=False)

print(f'Synthetic dataset generated with {len(df)} samples.')
print(f'Columns: {list(df.columns)}')
print(f'Loan Approval Distribution:')
print(df['Loan_Status'].value_counts())
print(f'\n Dataset saved to data/train.csv')
