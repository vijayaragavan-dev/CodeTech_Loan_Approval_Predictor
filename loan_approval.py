"""
=================================================================
LOAN APPROVAL PREDICTOR USING MACHINE LEARNING
=================================================================
A professional Machine Learning system that predicts whether a loan
application will be approved (Y) or rejected (N) based on applicant
information such as income, credit history, loan amount, etc.

This project demonstrates a complete end-to-end ML pipeline:
  1. Data Loading & Exploration
  2. Data Cleaning & Missing Value Handling
  3. Data Transformation & Encoding
  4. Exploratory Data Analysis & Visualization
  5. Feature Selection & Train-Test Splitting
  6. Model Training (Logistic Regression, Decision Tree, Random Forest)
  7. Model Evaluation & Comparison
  8. Best Model Saving for Future Predictions

Author  : Vijayaragavan U
College : Saranathan College of Engineering
=================================================================
"""

# ======================================================================
# STAGE 1: IMPORT LIBRARIES
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import io
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set Seaborn style and Matplotlib defaults for professional plots
sns.set_style('whitegrid')
plt.rcParams['figure.dpi']          = 150
plt.rcParams['figure.figsize']      = (10, 6)
plt.rcParams['savefig.dpi']         = 150

plt.rcParams['font.size']           = 12

# Create output directories if they do not exist
os.makedirs('data', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ======================================================================
# STAGE 2: LOAD DATASET
# ======================================================================

DATA_PATH = 'data/train.csv'

if not os.path.exists(DATA_PATH):
    print(f"[INFO] {DATA_PATH} not found. Generating synthetic dataset...")
    os.system(f'{sys.executable} generate_data.py')

print("=" * 60)
print("STAGE 2: LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\n[SUCCESS] Dataset loaded successfully from {DATA_PATH}")
print(f"[INFO] Dataset contains {df.shape[0]} rows and {df.shape[1]} columns\n")

# Introduce missing values artificially to demonstrate the cleaning pipeline.
# In a real-world scenario these would already exist in the data.
np.random.seed(42)
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History']:
    mask = np.random.random(size=len(df)) < 0.035
    df.loc[mask, col] = np.nan

# ======================================================================
# STAGE 3: DATASET EXPLORATION
# ======================================================================

print("=" * 60)
print("STAGE 3: DATASET EXPLORATION")
print("=" * 60)

# --- 3a. First 5 rows ---
print("\n[INFO] First 5 rows of the dataset:")
print(df.head())

# --- 3b. Dataset Shape ---
print(f"\n[INFO] Dataset Shape (Rows, Columns): {df.shape}")

# --- 3c. Data Types and Non-Null Counts ---
print("\n[INFO] Dataset Info (Data Types & Non-Null Counts):")
buffer = io.StringIO()
df.info(buf=buffer)
print(buffer.getvalue())

# --- 3d. Statistical Summary ---
print("\n[INFO] Statistical Summary of Numerical Columns:")
print(df.describe())

# --- 3e. Missing Value Analysis ---
print("\n[INFO] Missing Value Count per Column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values found.")

# --- 3f. Class Distribution ---
print("\n[INFO] Loan Status Distribution (Target Variable):")
print(df['Loan_Status'].value_counts())
print("\n[INFO] Loan Status Distribution (Percentage):")
print(df['Loan_Status'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# ======================================================================
# STAGE 4 & 5: DATA CLEANING & MISSING VALUE HANDLING
# ======================================================================

print("\n" + "=" * 60)
print("STAGES 4 & 5: DATA CLEANING & MISSING VALUE HANDLING")
print("=" * 60)

original_shape = df.shape

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove 'Loan_ID' from categorical columns (it is just an identifier, not a feature)
if 'Loan_ID' in categorical_cols:
    categorical_cols.remove('Loan_ID')
# Remove the target variable from feature imputation
if 'Loan_Status' in categorical_cols:
    categorical_cols.remove('Loan_Status')

print(f"\n[INFO] Numerical columns for imputation : {numerical_cols}")
print(f"[INFO] Categorical columns for imputation: {categorical_cols}")

# --- Numerical Missing Value Imputation (Median Strategy) ---
# Why median? Median is robust to outliers. Unlike the mean, it is not skewed
# by extreme values in income or loan amount, making it a safe choice.
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"[CLEANING] Filled missing values in '{col}' with median: {median_val}")

# --- Categorical Missing Value Imputation (Mode Strategy) ---
# Why mode? Mode is the most frequent category. For categorical data it is the
# safest single-value imputation because it preserves the majority distribution.
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"[CLEANING] Filled missing values in '{col}' with mode: '{mode_val}'")

print(f"\n[CLEANING COMPLETE] Dataset shape after cleaning: {df.shape}")

# ======================================================================
# STAGE 6: DATA TRANSFORMATION
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 6: DATA TRANSFORMATION")
print("=" * 60)

# Convert 'Dependents' 3+ to just 3 for numerical processing
df['Dependents'] = df['Dependents'].replace('3+', '3')

# Ensure LoanAmount and Loan_Amount_Term are numeric (they may have been read as objects)
df['LoanAmount'] = pd.to_numeric(df['LoanAmount'], errors='coerce')
df['Loan_Amount_Term'] = pd.to_numeric(df['Loan_Amount_Term'], errors='coerce')

# If any coercions introduced NaN, fill them with median/mode
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

# Convert Dependents to integer
df['Dependents'] = df['Dependents'].astype(int)

print("[TRANSFORMATION] Data transformations applied successfully.")

# --- Comprehensive NaN check after all transformations ---
remaining_nan = df.isnull().sum()
total_nan = remaining_nan.sum()
if total_nan > 0:
    print(f"\n[WARNING] {total_nan} NaN value(s) remaining after transformation. Performing final imputation:")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("[CLEANING] Final imputation complete.")
else:
    print("[TRANSFORMATION] No NaN values remain. Data is clean.")

# ======================================================================
# STAGE 7: LABEL ENCODING
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 7: LABEL ENCODING")
print("=" * 60)

# Identify categorical columns that need encoding (exclude Loan_ID and Loan_Status)
encode_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Loan_ID' in encode_cols:
    encode_cols.remove('Loan_ID')
if 'Loan_Status' in encode_cols:
    encode_cols.remove('Loan_Status')

print(f"\n[INFO] Columns to encode: {encode_cols}")

# Store label encoders for potential future inverse transformation
label_encoders = {}

for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"[ENCODING] '{col}' encoded -> Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode the target variable separately
target_encoder = LabelEncoder()
df['Loan_Status'] = target_encoder.fit_transform(df['Loan_Status'])
print(f"\n[ENCODING] Target 'Loan_Status' encoded -> Mapping: {dict(zip(target_encoder.classes_,
                                                                       target_encoder.transform(target_encoder.classes_)))}")

print("\n[INFO] First 5 rows after encoding:")
print(df.head())

# ======================================================================
# STAGE 8: EXPLORATORY DATA ANALYSIS (VISUALIZATIONS)
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 8: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- 8a. Target Variable Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
ax1 = axes[0]
counts = df['Loan_Status'].value_counts()
labels = ['Approved (Y)' if k == 1 else 'Rejected (N)' for k in counts.index]
bars = ax1.bar(labels, counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.2)
ax1.set_title('Loan Approval Distribution', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Number of Applicants', fontsize=12)
ax1.set_xlabel('Loan Status', fontsize=12)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             str(val), ha='center', fontweight='bold', fontsize=11)

# Pie chart
ax2 = axes[1]
colors_pie = ['#2ecc71', '#e74c3c']
ax2.pie(counts.values, labels=['Approved', 'Rejected'], autopct='%1.1f%%',
        startangle=90, colors=colors_pie, explode=(0.05, 0.05),
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Loan Approval Proportion', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('screenshots/target_distribution.png', dpi=150)
plt.close()
print("[VISUALIZATION] Target distribution saved to 'screenshots/target_distribution.png'")

# --- 8b. Correlation Heatmap ---
plt.figure(figsize=(14, 10))
# Compute correlation matrix on numerical features
corr_matrix = df[numerical_cols + ['Loan_Status']].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(240, 10, as_cmap=True)

sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
    center=0, square=True, linewidths=0.8,
    annot_kws={'size': 10, 'fontweight': 'bold'},
    cbar_kws={'shrink': 0.75, 'label': 'Correlation Coefficient'}
)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('screenshots/correlation_heatmap.png', dpi=150)
plt.close()
print("[VISUALIZATION] Correlation heatmap saved to 'screenshots/correlation_heatmap.png'")

# --- 8c. Additional EDA: Loan Status by Categorical Features ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

cat_features_eda = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
actual_cat_cols = [col for col in cat_features_eda if col in encode_cols]

for idx, col in enumerate(actual_cat_cols):
    row, col_idx = divmod(idx, 3)
    ax = axes[row, col_idx]
    crosstab = pd.crosstab(df[col], df['Loan_Status'], normalize='index') * 100
    crosstab.columns = ['Rejected (%)', 'Approved (%)']
    crosstab.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=0.8, legend=False)

    le = label_encoders.get(col)
    if le:
        tick_labels = le.classes_
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

    ax.set_title(f'Approval Rate by {col}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_xlabel(col, fontsize=11)
    ax.legend(['Rejected', 'Approved'], fontsize=10, loc='upper right')
    ax.set_ylim(0, 105)

if len(actual_cat_cols) < 6:
    for idx in range(len(actual_cat_cols), 6):
        row, col_idx = divmod(idx, 3)
        fig.delaxes(axes[row, col_idx])

plt.suptitle('Loan Approval Rate by Applicant Characteristics', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('screenshots/approval_by_category.png', dpi=150)
plt.close()
print("[VISUALIZATION] Approval-by-category chart saved to 'screenshots/approval_by_category.png'")

# ======================================================================
# STAGE 9: FEATURE SELECTION
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 9: FEATURE SELECTION")
print("=" * 60)

# Separate features (X) and target (y)
# Drop Loan_ID (identifier, not predictive) and Loan_Status (target)
feature_cols = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status']]

X = df[feature_cols]
y = df['Loan_Status']

print(f"\n[INFO] Selected Features ({len(feature_cols)}): {feature_cols}")
print(f"[INFO] Feature matrix shape: {X.shape}")
print(f"[INFO] Target vector shape : {y.shape}")

# ======================================================================
# STAGE 10: TRAIN-TEST SPLIT
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 10: TRAIN-TEST SPLIT")
print("=" * 60)

# Split data: 80% training, 20% testing
# stratify=y ensures the class distribution is preserved in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n[INFO] Training set   : {X_train.shape[0]} samples")
print(f"[INFO] Testing set    : {X_test.shape[0]} samples")
print(f"[INFO] Train class distribution:\n{y_train.value_counts().to_string()}")
print(f"[INFO] Test class distribution:\n{y_test.value_counts().to_string()}")

# ======================================================================
# STAGE 11: MODEL TRAINING
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 11: MODEL TRAINING")
print("=" * 60)

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Random Forest'      : RandomForestClassifier(random_state=42, n_jobs=-1)
}

trained_models = {}

for name, model in models.items():
    print(f"\n[Training] {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"[SUCCESS] {name} training complete.")

# ======================================================================
# STAGE 12: PREDICTION
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 12: PREDICTION")
print("=" * 60)

predictions = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    print(f"[PREDICTION] {name} predictions generated ({len(y_pred)} samples).")

# ======================================================================
# STAGE 13: MODEL EVALUATION
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 13: MODEL EVALUATION")
print("=" * 60)

results = []
best_model_name = None
best_f1 = -1

for name in models.keys():
    y_pred = predictions[name]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                    'Recall': rec, 'F1 Score': f1})

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected (N)', 'Approved (Y)']))

    # Confusion Matrix per model
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}")

    # Track best model by F1 Score
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

# Build a comparison DataFrame
results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)

print(f"\n{'=' * 60}")
print(f"  MODEL COMPARISON (Sorted by F1 Score)")
print(f"{'=' * 60}")
print(results_df.to_string(index=False))

print(f"\n{'=' * 60}")
print(f"  BEST MODEL: {best_model_name} (F1 Score: {best_f1:.4f})")
print(f"{'=' * 60}")

# --- Save Confusion Matrix of Best Model ---
best_model = trained_models[best_model_name]
y_pred_best = predictions[best_model_name]
cm_best = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Rejected (N)', 'Approved (Y)'],
            yticklabels=['Rejected (N)', 'Approved (Y)'],
            annot_kws={'size': 14, 'fontweight': 'bold'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('screenshots/confusion_matrix.png', dpi=150)
plt.close()
print(f"\n[VISUALIZATION] Confusion matrix saved to 'screenshots/confusion_matrix.png'")

# ======================================================================
# SAVE MODEL METRICS TO FILE
# ======================================================================

with open('outputs/model_metrics.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("  LOAN APPROVAL PREDICTOR - MODEL EVALUATION METRICS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Date              : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Test set size     : {X_test.shape[0]} samples\n")
    f.write(f"Training set size : {X_train.shape[0]} samples\n")
    f.write(f"Features used     : {len(feature_cols)}\n\n")
    f.write("-" * 70 + "\n")
    f.write("  MODEL PERFORMANCE COMPARISON\n")
    f.write("-" * 70 + "\n\n")

    for _, row in results_df.iterrows():
        f.write(f"  Model      : {row['Model']}\n")
        f.write(f"  Accuracy   : {row['Accuracy']:.4f}\n")
        f.write(f"  Precision  : {row['Precision']:.4f}\n")
        f.write(f"  Recall     : {row['Recall']:.4f}\n")
        f.write(f"  F1 Score   : {row['F1 Score']:.4f}\n")
        f.write("  " + "-" * 40 + "\n\n")

    f.write(f"\n{'=' * 70}\n")
    f.write(f"  BEST MODEL: {best_model_name}\n")
    f.write(f"{'=' * 70}\n\n")

    f.write("  Classification Report:\n\n")
    f.write(classification_report(y_test, y_pred_best,
                                   target_names=['Rejected (N)', 'Approved (Y)']))
    f.write("\n  Confusion Matrix:\n")
    f.write(f"  {cm_best[0]}\n")
    f.write(f"  {cm_best[1]}\n")

print(f"\n[OUTPUT] Model metrics saved to 'outputs/model_metrics.txt'")

# ======================================================================
# SAVE DATASET SUMMARY TO FILE
# ======================================================================

with open('outputs/dataset_summary.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("  LOAN APPROVAL PREDICTOR - DATASET SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset Shape        : {df.shape[0]} rows, {df.shape[1]} columns\n")
    f.write(f"Number of Features   : {len(feature_cols)}\n")
    f.write(f"Target Variable      : Loan_Status\n\n")

    # Original data info
    f.write("-" * 70 + "\n")
    f.write("  DATA TYPES\n")
    f.write("-" * 70 + "\n")
    f.write(df.dtypes.to_string() + "\n\n")

    f.write("-" * 70 + "\n")
    f.write("  STATISTICAL SUMMARY (Numerical Features)\n")
    f.write("-" * 70 + "\n")
    f.write(df[numerical_cols + ['Loan_Status']].describe().to_string() + "\n\n")

    f.write("-" * 70 + "\n")
    f.write("  MISSING VALUES (Before Cleaning)\n")
    f.write("-" * 70 + "\n")
    # Reload original data to show missing values pre-cleaning
    df_raw = pd.read_csv(DATA_PATH)
    missing_str = df_raw.isnull().sum().to_string()
    f.write(missing_str + "\n")

    # Mark which were artificially added
    f.write("\n  (Note: Some missing values were artificially introduced\n")
    f.write("   to demonstrate the data cleaning pipeline. Real data\n")
    f.write("   from production would have similar missing entries.)\n\n")

    f.write("-" * 70 + "\n")
    f.write("  CLASS DISTRIBUTION\n")
    f.write("-" * 70 + "\n")
    dist = df_raw['Loan_Status'].value_counts()
    for status, count in dist.items():
        pct = count / len(df_raw) * 100
        f.write(f"  {status:>10} : {count:>4} ({pct:.2f}%)\n")

print(f"[OUTPUT] Dataset summary saved to 'outputs/dataset_summary.txt'")

# ======================================================================
# STAGE 14: MODEL SAVING (Pickle)
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 14: MODEL SAVING")
print("=" * 60)

MODEL_PATH = 'loan_model.pkl'

model_package = {
    'model'          : best_model,
    'features'       : feature_cols,
    'label_encoders' : label_encoders,
    'target_encoder' : target_encoder,
    'model_name'     : best_model_name,
    'metrics'        : results_df.to_dict('records')
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n[SUCCESS] Best model saved to '{MODEL_PATH}'")
print(f"[INFO]    Model package includes: model object, feature list, encoders, and metrics.")

# ======================================================================
# STAGE 15: PROJECT COMPLETION SUMMARY
# ======================================================================

print("\n" + "=" * 60)
print("STAGE 15: PROJECT COMPLETION SUMMARY")
print("=" * 60)

print(f"""
{"=" * 60}
  LOAN APPROVAL PREDICTOR - PROJECT COMPLETED SUCCESSFULLY
{"=" * 60}

  Pipeline Stages Completed:
  -------------------------
  1.  Libraries Imported     - pandas, numpy, matplotlib, seaborn, sklearn
  2.  Dataset Loaded          - {df.shape[0]} samples from {DATA_PATH}
  3.  Dataset Explored        - shape, info, stats, missing values, class distribution
  4.  Data Cleaned            - handled missing values
  5.  Missing Values Handled  - median (numerical), mode (categorical)
  6.  Data Transformed        - type conversions, categorical mapping
  7.  Label Encoding Applied  - all categorical features encoded
  8.  EDA & Visualizations    - distribution, heatmap, category analysis
  9.  Features Selected       - {len(feature_cols)} predictive features
  10. Train-Test Split        - 80% train, 20% test
  11. Models Trained          - {', '.join(models.keys())}
  12. Predictions Generated   - evaluated on held-out test set
  13. Models Evaluated        - accuracy, precision, recall, f1, confusion matrix
  14. Best Model Saved        - {best_model_name} -> '{MODEL_PATH}'
  15. Summary Generated       - metrics saved to outputs/

  Best Model: {best_model_name}
  F1 Score  : {best_f1:.4f}

  Output Files Generated:
  ----------------------
  screenshots/target_distribution.png
  screenshots/correlation_heatmap.png
  screenshots/confusion_matrix.png
  screenshots/approval_by_category.png
  outputs/model_metrics.txt
  outputs/dataset_summary.txt
  loan_model.pkl

  Author : Vijayaragavan U
  Project: CodeTech IT Solutions Internship
{"=" * 60}
""")
