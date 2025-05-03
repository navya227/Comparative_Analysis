# Comparative Analysis of Explainability in Machine Learning Algorithms

This repository presents an in-depth comparative analysis of machine learning models across classification and regression tasks with a strong emphasis on **explainability** using **LIME** and **SHAP**. The study evaluates both predictive performance and interpretability of models on three benchmark datasets: **Pima Diabetes**, **Communities & Crime**, and **Breast Cancer**.

---

##  Objectives

- Evaluate explainability of ML models using LIME and SHAP.
- Compare model performance across classification and regression tasks.
- Analyze fidelity and sparsity of explanations to assess robustness.
- Recommend models for high-stakes domains requiring both accuracy and interpretability.

---

##  Datasets Used

1. **Pima Indians Diabetes Dataset** – Binary classification of diabetic cases using 8 medical attributes.
2. **Communities and Crime Dataset** – Regression task to predict violent crime rates from 128 socio-economic indicators.
3. **Breast Cancer Wisconsin Dataset** – Classification of tumors as malignant or benign using cell image features.

---

##  Machine Learning Models

### Classification:
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Neural Network (MLP)
- XGBoost

### Regression:
- Linear Regression
- Decision Tree Regressor
- Support Vector Regressor (SVR)
- Random Forest Regressor
- XGBoost Regressor

---

##  Explainability Methods

- **LIME (Local Interpretable Model-Agnostic Explanations)**: Generates local linear approximations to explain individual predictions.
- **SHAP (SHapley Additive exPlanations)**: Computes feature contributions using Shapley values for global and local interpretability.

---

##  Evaluation Metrics

### Classification:
- Accuracy, Precision, Recall, F1-Score, AUC-ROC  
- LIME Fidelity & Sparsity, SHAP Sparsity

### Regression:
- MAE, MSE, RMSE, R² Score  
- SHAP Sparsity

---

## Setup Instructions

### Requirements

- Python 3.8+
- scikit-learn
- xgboost
- shap
- lime
- pandas, numpy, matplotlib

### Installation

```bash
pip install -r requirements.txt
# or install manually:
pip install scikit-learn xgboost shap lime pandas numpy matplotlib
```

---

##  How to Run

1. **Preprocess data** (Crime):
```bash
python dataloader.py
```

2. **Run classification (Pima, Breast Cancer):**
```bash
python classification.py
```

3. **Run regression (Crime):**
```bash
python regression.py
```

4. **Logistic regression analysis (Pima-specific):**
```bash
python logistic.py
```

> Outputs including model metrics, LIME explanations, SHAP plots, and CSV results are saved in `../results/` subdirectories.

---
