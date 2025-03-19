import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss

# Step 1: Load the training and test data
train_file_path = 'simCRdata.csv'  # Replace with your actual training file path
test_file_path = 'simCRdata_test.csv'    # Replace with your actual test file path

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Step 2: Prepare the target variable (y_next) for both training and test data
train_data['y_next'] = train_data.groupby('cust')['y'].shift(-1)
test_data['y_next'] = test_data.groupby('cust')['y'].shift(-1)

# Drop rows where 'y_next' is NaN (i.e., the last row for each customer)
train_data_cleaned = train_data.dropna(subset=['y_next'])
test_data_cleaned = test_data.dropna(subset=['y_next'])

# Step 3: One-hot encode 'y' (current status) and optionally 't' (time step) for input
# Use pd.get_dummies() to one-hot encode the 'y' column (current credit status)
X_train = pd.get_dummies(train_data_cleaned[['y']], prefix='y')
X_test = pd.get_dummies(test_data_cleaned[['y']], prefix='y')
print(X_test.head())

# If you want to include 't' as well, you can one-hot encode it like this:
# X_train_t = pd.get_dummies(train_data_cleaned[['t']], prefix='t')
# X_test_t = pd.get_dummies(test_data_cleaned[['t']], prefix='t')

# Optional: If including 't', concatenate it with the one-hot encoded 'y'
# X_train = pd.concat([X_train, X_train_t], axis=1)
# X_test = pd.concat([X_test, X_test_t], axis=1)

# Step 4: Define the target variable (y_next)
y_train = train_data_cleaned['y_next']
y_test = test_data_cleaned['y_next']
print(y_test.head())

# Step 5: Initialize and train the MLPClassifier (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(2,), solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Step 6: Predict on the test data
y_pred = mlp.predict(X_test)
print("result:", y_pred)

# Step 7: Evaluate the model performance using classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Output the predicted probabilities for the next state
y_pred_proba = mlp.predict_proba(X_test)

# Display predicted probabilities for the first few test samples
print("Predicted Probabilities for First 5 Samples:")
print(y_pred_proba[:5])
#######brier score

# Step 2: Convert the true labels (y_test) to one-hot format
# You can use pd.get_dummies or manually create a one-hot encoded matrix
y_test_one_hot = pd.get_dummies(y_test).values

# Step 3: Calculate the Brier score for each class
# Initialize an empty list to store the Brier scores for each class
brier_scores = []

# Iterate over each class to calculate the Brier score
for i in range(y_pred_proba.shape[1]):  # Loop over the number of classes
    brier_score = brier_score_loss(y_test_one_hot[:, i], y_pred_proba[:, i])
    brier_scores.append(brier_score)

# Print the Brier scores for each class
for i, score in enumerate(brier_scores):
    print(f"Brier score for class {i}: {score}")

# Calculate the average Brier score across all classes
average_brier_score = np.mean(brier_scores)
print(f"Average Brier score: {average_brier_score}")