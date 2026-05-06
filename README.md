# Early Alzheimer's Detection via Handwriting

This project predicts early signs of Alzheimer's disease from handwriting 
benchmark tests using a machine learning ensemble pipeline.

## Problem

Early Alzheimer's detection from handwriting data is a non-invasive 
diagnostic approach. The DARWIN dataset contains handwriting task features 
from Alzheimer's patients and healthy controls. The goal was to build an 
ensemble model that reduces feature redundancy via PCA while maximizing 
predictive accuracy.

## Approach

- Dimensionality reduction via PCA (65 features retained)
- Ensemble voting classifier combining Random Forest, LinearSVC, XGBoost
- 5-fold cross-validation for model selection
- Evaluation across accuracy, precision, recall, specificity, and F1-score

## Results (Reduced Feature Set 450 -> 65)

| Model | Accuracy | F1-Score | Mean CV Score |
|-------|----------|----------|---------------|
| **Ensemble** | **94.3%** | **95.0%** | **75.6%** |
| Random Forest | 91.4% | 92.7% | 77.0% |
| LinearSVC | 91.4% | 92.3% | 69.0% |
| XGBoost | 88.6% | 90.0% | 67.6% |
| Logistic Regression | 88.6% | 89.5% | 73.4% |
| GaussianNB | 80.0% | 84.4% | 75.6% |
| SVM | 82.9% | 85.0% | 81.3% |

The ensemble outperformed all individual models on accuracy and F1-score.

## Tech Stack

- Python 3.11+
- scikit-learn
- pandas, numpy
- XGBoost, LightGBM
- matplotlib / seaborn

## Folder Structure

early-alzheimers-detection/
├─ src/
│   └─ gp2_ensemble_model_darwin_dataset.py
├─ data/
│   └─ data.csv
├─ collab notebook/
│   └─ GP2_Ensemble_Model_DARWIN_Dataset.ipynb
├─ results/
│   └─ PCA_Results.xlsx
├─ README.md
├─ requirements.txt
└─ .gitignore
