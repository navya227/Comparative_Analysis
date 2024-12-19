import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Breast Cancer Dataset from the .data file without headers
data = pd.read_csv('../data/crime/communities.data', header=None)
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
# Split the dataset into features (X) and target (y)
X = data.iloc[:, 5:127]  # select all columns from index 2 onwards as features
y = data.iloc[:, 127]   # select the second column (index 1) as the target variable
# print(y)
# Perform the train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
# Scale the features for consistency across models
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Save the train-test split to CSV files for reuse
X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(1, X.shape[1] + 1)])
X_test_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(1, X.shape[1] + 1)])
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)
print(y_train_df)
X_train.to_csv('../data/crime/X_train.csv', index=False)
X_test.to_csv('../data/crime/X_test.csv', index=False)
y_train.to_csv('../data/crime/y_train.csv', index=False)
y_test.to_csv('../data/crime/y_test.csv', index=False)

print("Breast Cancer dataset has been preprocessed and saved to CSV files.")
