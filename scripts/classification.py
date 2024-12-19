import os
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Define output directory
output_dir = "../results/pima"
os.makedirs(output_dir, exist_ok=True)

# Load the preprocessed train-test data
X_train = pd.read_csv('../data/pima/X_train.csv')
X_test = pd.read_csv('../data/pima/X_test.csv')
y_train = pd.read_csv('../data/pima/y_train.csv').values.ravel()  # flatten to 1D array
y_test = pd.read_csv('../data/pima/y_test.csv').values.ravel()

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Neural Network": MLPClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Initialize results storage
results = []

def evaluate_model(model, model_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Save results to text file
    with open(f"{output_dir}/model_performance.txt", "a") as f:
        f.write(f"\n{model_name} Model Performance:\n")
        f.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}\n")

    # SHAP analysis
    if model_name in ["Random Forest", "XGBoost"]:
        explainer_shap = shap.TreeExplainer(model, data=X_train)
        shap_values = explainer_shap(X_test, check_additivity=False)
    elif model_name in ["SVM", "Neural Network"]:
        explainer_shap = shap.KernelExplainer(model.predict, X_train.sample(100, random_state=42))  # sampling to reduce computation
        shap_values = explainer_shap.shap_values(X_test)
    else:
        explainer_shap = shap.Explainer(model, X_train)
        shap_values = explainer_shap(X_test)

    # Save SHAP summary plot
    shap_summary_path = f"{output_dir}/{model_name}_shap_summary.png"
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, show=False)
    plt.savefig(shap_summary_path)
    plt.close()

    # LIME analysis and save explanation as text
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['No Diabetes', 'Diabetes'], mode='classification')
    exp = explainer_lime.explain_instance(X_test.values[0], model.predict_proba)
    
    # Save LIME explanation to a text file
    lime_text_path = f"{output_dir}/{model_name}_lime_explanation.txt"
    with open(lime_text_path, "w") as f:
        f.write(f"LIME Explanation for {model_name} (Instance 0):\n")
        f.write(exp.as_list().__str__())  # Write explanation in text format

    # Fidelity and sparsity for LIME
    lime_fidelity = exp.score
    lime_sparsity = len(exp.local_exp[1])
    
    # SHAP fidelity and sparsity
    if model_name in ["SVM", "Neural Network"]:
        shap_fidelity = np.mean(np.abs(shap_values).sum(axis=1))
        shap_sparsity = np.count_nonzero(np.abs(shap_values).mean(axis=0))
    else:
        shap_fidelity = np.mean(np.abs(shap_values.values).sum(axis=1))
        shap_sparsity = np.count_nonzero(np.abs(shap_values.values).mean(axis=0))

    # Save metrics to results list
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC-ROC": auc_roc,
        "LIME Fidelity": lime_fidelity,
        "LIME Sparsity": lime_sparsity,
        "SHAP Fidelity": shap_fidelity,
        "SHAP Sparsity": shap_sparsity
    })

# Run evaluation for each model
for model_name, model in models.items():
    evaluate_model(model, model_name)

# Save results to a DataFrame and export to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/model_explainability_results.csv", index=False)

# Also write the results summary to a text file
with open(f"{output_dir}/model_explainability_summary.txt", "w") as f:
    f.write("Model Explainability and Performance Comparison:\n\n")
    f.write(results_df.to_string(index=False))
    
print(f"Results saved in '{output_dir}' directory.")
