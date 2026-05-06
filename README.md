# 🧠 Alzheimer’s Detection using Machine Learning

## 📌 Overview
This project focuses on detecting early signs of Alzheimer’s disease using machine learning techniques applied on handwriting-based features from the DARWIN dataset.

The goal is to build an accurate and robust classification system using an ensemble approach.

---

## 🎯 Objective
To develop a machine learning model that can classify individuals as Alzheimer’s patients or healthy controls based on extracted handwriting features.

---

## ⚙️ Approach

- Applied Principal Component Analysis (PCA) for dimensionality reduction (450 → 65 features)
- Built an Ensemble Voting Classifier combining:
  - Random Forest
  - Linear SVM (LinearSVC)
  - XGBoost
- Used 5-Fold Cross Validation for evaluation
- Compared multiple ML models

---

## 📊 Results

The ensemble model achieved the best performance compared to individual models in terms of accuracy and F1-score.

---

## 🧰 Tech Stack

- Python
- Scikit-learn
- Pandas & NumPy
- XGBoost
- Matplotlib / Seaborn

---

## 👨‍💻 My Contribution

- Worked on the machine learning model
- Applied PCA for feature reduction
- Built and tested ensemble models
- Evaluated model performance using multiple metrics
