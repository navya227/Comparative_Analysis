# train_logistic_regression.py

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the preprocessed train-test data from CSV files
X_train = pd.read_csv('../data/pima_diabetes/X_train_pima.csv')
X_test = pd.read_csv('../data/pima_diabetes/X_test_pima.csv')
y_train = pd.read_csv('../data/pima_diabetes/y_train_pima.csv').values.ravel()  # flatten to 1D array
y_test = pd.read_csv('../data/pima_diabetes/y_test_pima.csv').values.ravel()    # flatten to 1D array

# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# ------------------------
# SHAP Explanation
# ------------------------

# Create a SHAP explainer object (for linear models like logistic regression, LinearExplainer works best)
explainer_shap = shap.LinearExplainer(lr_model, X_train)
shap_values = explainer_shap.shap_values(X_test)

# Summary plot of SHAP values
shap.summary_plot(shap_values, X_test, feature_names=X_train.columns)

# Force plot for a specific instance
shap.force_plot(explainer_shap.expected_value, shap_values[0], X_test.iloc[0], feature_names=X_train.columns)

# ------------------------
# LIME Explanation
# ------------------------

# Create LIME explainer object
explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['No Diabetes', 'Diabetes'], mode='classification')

# Explain a single instance using LIME
i = 0  # index of the instance to explain
exp = explainer_lime.explain_instance(X_test.values[i], lr_model.predict_proba)

# Visualize the LIME explanation for the first test instance
exp.show_in_notebook()

# LIME quantitative evaluation (fidelity of local explanations)
lime_explanation = exp.local_exp[1]  # feature contributions for class '1' (Diabetes)
print(f"LIME Explanation for instance {i}:")
for feature, weight in lime_explanation:
    print(f"Feature {X_train.columns[feature]}: {weight}")

# ------------------------
# Quantitative Evaluation of Explainability (LIME and SHAP)
# ------------------------

# Fidelity for LIME (how well LIME approximates the model locally)
lime_fidelity = exp.score
print(f"LIME Fidelity for instance {i}: {lime_fidelity:.4f}")

# Sparsity of explanations (fewer features used = more sparse)
lime_sparsity = len(exp.local_exp[1])  # Number of features used in the local explanation
print(f"LIME Sparsity (number of features in explanation): {lime_sparsity}")

# SHAP Fidelity and Stability
# SHAP is inherently stable and accurate (based on Shapley values), so you can evaluate the feature importance
shap_fidelity = np.mean(np.abs(shap_values).sum(axis=1))  # Mean SHAP contribution per feature
print(f"Mean SHAP feature contribution: {shap_fidelity:.4f}")

# SHAP sparsity: SHAP gives values for all features, but you can control how many features are shown
shap_sparsity = np.count_nonzero(np.abs(shap_values).mean(axis=0))
print(f"SHAP Sparsity (number of features with non-zero importance): {shap_sparsity}")

# Results comparison for Logistic Regression (LIME vs SHAP)
print(f"\nExplainability Results for Logistic Regression:")
print(f"LIME Fidelity: {lime_fidelity:.4f}, LIME Sparsity: {lime_sparsity}")
print(f"SHAP Sparsity: {shap_sparsity}, SHAP Average Feature Contribution: {shap_fidelity:.4f}")
