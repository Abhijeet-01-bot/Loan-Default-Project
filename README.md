# 🏦 Loan Default Prediction System

## 📌 Project Overview
This project predicts whether a loan applicant is likely to default or not using Machine Learning techniques.

It includes:
- End-to-end ML pipeline
- Model comparison
- FastAPI deployment
- MLflow for experiment tracking

---

## 🚀 Features
✅ Data preprocessing (handling missing values, encoding)  
✅ Class imbalance handling using SMOTE  
✅ Multiple ML models trained and compared  
✅ Best model selection (Naive Bayes)  
✅ REST API using FastAPI  
✅ MLflow integration for experiment tracking  

---

## 📊 Dataset
- Contains loan application data  
- Includes features like:
  - Income  
  - Loan amount  
  - Credit score  
  - Age  
  - Debt-to-income ratio  

---

## 🧠 Machine Learning Models Used

| Model | Accuracy |
|------|--------|
| Logistic Regression | 0.72 |
| Decision Tree | 0.75 |
| Random Forest | 0.75 |
| Naive Bayes ✅ | 0.91 |
| KNN | 0.89 |

✅ **Best Model: Naive Bayes (Accuracy ~91%)**

---

## ⚙️ Project Structure
Loan-Default-Project/
│
├── data/
│   └── Loan_Default.csv
│
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── api/
│   └── app.py
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│
├── MLworkflow/
│   └── mlflow_script.py
│
├── mlruns/                 ✅ (auto-created by MLflow)
│
├── venv/                   ✅ (your virtual environment)
│
├── requirements.txt        ✅ (frozen dependencies)
│
├── README.md               ✅ (GitHub documentation)
