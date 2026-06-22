# 🏦 Loan Approval Predictor Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Internship](https://img.shields.io/badge/CodeTech-Internship-blueviolet)

---

## Internship Information

| Field              | Details                         |
| ------------------ | ------------------------------- |
| Internship Program | Machine Learning Internship     |
| Organization       | CodeTech IT Solutions Pvt. Ltd. |
| Intern Name        | Vijayaragavan U                 |
| Intern ID          | CITS5452                       |
| Duration           | 4 Weeks                         |
| Internship Period  | 18 June 2026 – 16 July 2026     |
| Domain             | Machine Learning                |
| Project            | Loan Approval Predictor         |
| Project Type       | Banking Analytics               |

---

## 📖 Project Overview

The **Loan Approval Predictor** is a Machine Learning-based Banking Analytics project developed to predict whether a loan application is likely to be approved or rejected based on applicant information.

Financial institutions receive thousands of loan applications every day. Evaluating each application manually can be time-consuming, inconsistent, and prone to human error. Machine Learning enables banks to automate decision-making by analyzing historical applicant data and identifying patterns that influence loan approval outcomes.

This project uses classification algorithms to analyze applicant demographics, financial information, and credit history to predict loan approval status accurately.

---

## 🎯 Problem Statement

Loan approval is one of the most critical processes in the banking industry. Banks must minimize financial risk while ensuring deserving applicants receive loans.

The objective of this project is to build a Machine Learning model capable of predicting whether a loan application will be:

* ✅ Approved
* ❌ Rejected

based on customer information and financial indicators.

---

## 🚀 Project Objectives

* Perform exploratory data analysis on loan application data.
* Handle missing values using appropriate preprocessing techniques.
* Transform categorical variables into numerical representations.
* Visualize important relationships within the dataset.
* Train multiple Machine Learning classification models.
* Compare model performance using evaluation metrics.
* Select and save the best-performing model.
* Generate automated reports and visualizations.

---

## 📊 Dataset Information

### Dataset Used

Loan Approval Prediction Dataset

### Dataset Description

The dataset contains information about applicants, their financial background, education level, credit history, and property details.

### Features

| Feature           | Description                 |
| ----------------- | --------------------------- |
| Gender            | Applicant Gender            |
| Married           | Marital Status              |
| Dependents        | Number of Dependents        |
| Education         | Education Qualification     |
| Self_Employed     | Employment Status           |
| ApplicantIncome   | Applicant Monthly Income    |
| CoapplicantIncome | Co-Applicant Monthly Income |
| LoanAmount        | Requested Loan Amount       |
| Loan_Amount_Term  | Loan Repayment Duration     |
| Credit_History    | Credit Record Status        |
| Property_Area     | Urban / Semiurban / Rural   |
| Loan_Status       | Loan Approval Status        |

### Target Variable

| Value | Meaning       |
| ----- | ------------- |
| Y     | Loan Approved |
| N     | Loan Rejected |

---

## 🛠️ Technologies Used

| Category             | Technology          |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Processing      | Pandas, NumPy       |
| Data Visualization   | Matplotlib, Seaborn |
| Machine Learning     | Scikit-Learn        |
| Model Persistence    | Pickle              |
| IDE                  | Visual Studio Code  |
| Version Control      | Git & GitHub        |

---

## 🧠 Machine Learning Workflow

### 1. Data Collection

Load the loan approval dataset into a Pandas DataFrame.

### 2. Data Exploration

Analyze:

* Dataset Shape
* Feature Information
* Data Types
* Statistical Summary
* Target Distribution

### 3. Missing Value Handling

Missing values are handled using:

#### Numerical Features

* Median Imputation

#### Categorical Features

* Mode Imputation

This ensures data quality before training.

### 4. Data Transformation

Categorical features are encoded into numerical values using Label Encoding.

Examples:

* Gender
* Education
* Married
* Self_Employed
* Property_Area
* Loan_Status

### 5. Exploratory Data Analysis

Generate visualizations such as:

* Loan Approval Distribution
* Correlation Heatmap
* Approval by Category Analysis
* Confusion Matrix

### 6. Feature Selection

Identify relevant attributes influencing loan approval decisions.

### 7. Train-Test Split

Dataset divided into:

* 80% Training Data
* 20% Testing Data

### 8. Model Training

Train multiple Machine Learning models.

### 9. Model Evaluation

Evaluate models using:

* Accuracy
* Precision
* Recall
* F1 Score

### 10. Best Model Selection

Automatically select the best-performing model.

### 11. Model Saving

Save the final trained model for future predictions.

---

## 🤖 Machine Learning Algorithms Used

### Logistic Regression

A statistical classification algorithm used to predict binary outcomes such as loan approval or rejection.

### Decision Tree Classifier

A tree-based model that learns decision rules from applicant information.

### Random Forest Classifier

An ensemble learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## 📈 Model Performance

### Best Performing Model

**Logistic Regression**

### Evaluation Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 79.67% |
| Precision | 79.81% |
| Recall    | 95.40% |
| F1 Score  | 86.91% |

The model demonstrates strong performance in identifying approved loan applications while maintaining reliable overall classification capability.

---

## 📂 Project Structure

```text
CodeTech_Loan_Approval_Predictor/
│
├── data/
│   └── train.csv
│
├── screenshots/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── approval_by_category.png
│
├── outputs/
│   ├── dataset_summary.txt
│   └── model_metrics.txt
│
├── loan_model.pkl
├── loan_approval.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide

### Step 1: Clone Repository

```bash
git clone https://github.com/vijayaragavan-dev/CodeTech_Loan_Approval_Predictor.git
```

### Step 2: Navigate to Project Folder

```bash
cd CodeTech_Loan_Approval_Predictor
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Project

```bash
python loan_approval.py
```

---

## 📌 Outputs Generated

### Model File

```text
loan_model.pkl
```

### Reports

```text
outputs/dataset_summary.txt
outputs/model_metrics.txt
```

### Visualizations

```text
screenshots/target_distribution.png
screenshots/correlation_heatmap.png
screenshots/confusion_matrix.png
screenshots/approval_by_category.png
```

---

## 📷 Project Visualizations

### Loan Approval Distribution

![Loan Approval Distribution](screenshots/target_distribution.png)

### Correlation Heatmap

![Correlation Heatmap](screenshots/correlation_heatmap.png)

### Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

---

## 💼 Business Impact

This project demonstrates how Machine Learning can support banking institutions by:

* Reducing manual loan evaluation effort.
* Improving consistency in loan approval decisions.
* Identifying high-risk applications.
* Enhancing operational efficiency.
* Supporting data-driven financial decision-making.

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

* Data Cleaning
* Missing Value Handling
* Label Encoding
* Exploratory Data Analysis
* Feature Engineering
* Classification Algorithms
* Model Comparison
* Model Evaluation
* Business Analytics
* Technical Documentation

---

## 🔮 Future Enhancements

* Hyperparameter Tuning
* XGBoost Integration
* Streamlit Dashboard
* Flask REST API
* Real-Time Loan Eligibility Prediction
* Cloud Deployment

---

## 👨‍💻 Author

### Vijayaragavan U

Bachelor of Engineering (B.E.) – Computer Science and Engineering

Saranathan College of Engineering

Tiruchirappalli, Tamil Nadu, India

### Internship Details

* Organization: CodeTech IT Solutions Pvt. Ltd.
* Internship Domain: Machine Learning
* Intern ID: CITS4915
* Duration: 4 Weeks

### Connect With Me

* GitHub: https://github.com/vijayaragavan-dev
* LinkedIn: https://www.linkedin.com/in/vijaya-ragavan-ki10052007
* Portfolio: https://vijayaragavan.vercel.app

---

### ⭐ If you found this project useful, consider giving it a star on GitHub.

**Submitted as part of the Machine Learning Internship at CodeTech IT Solutions Pvt. Ltd. (Intern ID: CITS4915).**
