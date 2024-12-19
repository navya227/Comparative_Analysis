import os
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Define output directory
output_dir = "../results/crime"
os.makedirs(output_dir, exist_ok=True)

# Load the preprocessed train-test data
X_train = pd.read_csv('../data/crime/X_train.csv')
X_test = pd.read_csv('../data/crime/X_test.csv')
y_train = pd.read_csv('../data/crime/y_train.csv').values.ravel()  # flatten to 1D array
y_test = pd.read_csv('../data/crime/y_test.csv').values.ravel()

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "XGBoost Regressor": XGBRegressor(random_state=42)
}

# Initialize results storage
results = []

def evaluate_model(model, model_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Save results to text file
    with open(f"{output_dir}/model_performance.txt", "a") as f:
        f.write(f"\n{model_name} Model Performance:\n")
        f.write(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}\n")

    sample_size = min(100, X_train.shape[0])

# SHAP analysis
    if model_name in ["Random Forest Regressor", "XGBoost Regressor"]:
        explainer_shap = shap.TreeExplainer(model, data=X_train)
        shap_values = explainer_shap(X_test, check_additivity=False)
    elif model_name == "SVR":
        # Use KernelExplainer with adjusted sample size
        explainer_shap = shap.KernelExplainer(model.predict, X_train.sample(sample_size, random_state=42)) 
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
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, mode='regression')
    exp = explainer_lime.explain_instance(X_test.values[0], model.predict)
    
    # Save LIME explanation to a text file
    lime_text_path = f"{output_dir}/{model_name}_lime_explanation.txt"
    with open(lime_text_path, "w") as f:
        f.write(f"LIME Explanation for {model_name} (Instance 0):\n")
        f.write(exp.as_list().__str__())  # Write explanation in text format

    # SHAP sparsity
    if model_name == "SVR":
        shap_sparsity = np.count_nonzero(np.abs(shap_values).mean(axis=0))
    else:
        shap_sparsity = np.count_nonzero(np.abs(shap_values.values).mean(axis=0))

    # Save metrics to results list
    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
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
