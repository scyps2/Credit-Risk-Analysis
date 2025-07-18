import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from utils import *
from evaluations import *

MONTH_AHEAD = 6

df_performance = pd.read_csv('2020Q1_standard_performance.csv').head(500000)
df_origination = pd.read_csv('2020Q1_standard_origination.csv').head(500000)

df_performance = df_performance[[
    "Loan Sequence Number", "Monthly Reporting Period", 
    "Current Loan Delinquency Status"
]]
df_origination = df_origination[["Loan Sequence Number", "Credit Score"]]
df = pd.merge(df_origination, df_performance, on="Loan Sequence Number", how="inner")
print(df.head(), df.shape)

df = reclassify(df)
df = preprocess(df)
print(df.head(), df.shape)

columns_to_encode = ["Current Loan Delinquency Status", 'Next Loan Delinquency Status']
features = ['Credit Score']
encoded_columns, train_df, test_df, X_train, y_train, X_test, y_test = encode_and_split_loan(df, features)

# separate dataset by previous state
train_zero = train_df[train_df["Current Loan Delinquency Status"] == 0]
X_train_zero = train_zero[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')] + features]
y_train_zero = train_zero[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]                                                                    
test_zero = test_df[test_df["Current Loan Delinquency Status"] == 0]
X_test_zero = test_zero[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')] + features]
y_test_zero = test_zero[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]

train_nonzero = train_df[train_df["Current Loan Delinquency Status"] > 0]
X_train_nonzero = train_nonzero[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')] + features]
y_train_nonzero = train_nonzero[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]
test_nonzero = test_df[test_df["Current Loan Delinquency Status"] > 0]
X_test_nonzero = test_nonzero[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')] + features]
y_test_nonzero = test_nonzero[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]


# MLP Classifying
mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 5000, random_state = 1,
                   learning_rate_init = 0.0001, learning_rate = 'adaptive')

mlp.fit(X_train, y_train)
# y_pred_proba = mlp.predict_proba(X_test)
y_pred_proba = predict_n_months(mlp, MONTH_AHEAD, X_test)
# y_pred_proba = predict_n_months_weighted(mlp, MONTH_AHEAD, X_test)

mlp.fit(X_train_zero, y_train_zero)
# y_pred_proba_zero = mlp.predict_proba(X_test_zero)
y_pred_proba_zero = predict_n_months(mlp, MONTH_AHEAD, X_test_zero)
# y_pred_proba_zero = predict_n_months_weighted(mlp, MONTH_AHEAD, X_test_zero)

mlp.fit(X_train_nonzero, y_train_nonzero)
# y_pred_proba_nonzero = mlp.predict_proba(X_test_nonzero)
y_pred_proba_nonzero = predict_n_months(mlp, MONTH_AHEAD, X_test_nonzero)
# y_pred_proba_nonzero = predict_n_months_weighted(mlp, MONTH_AHEAD, X_test_nonzero)

# decode into integer columns(before one-hot)
current_states = np.argmax( 
    X_test[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')]].values,
    axis=1
)
T = transition_matrix(current_states, y_pred_proba)
print(f"Transition Matrix:{T}\n")
plot_transition_heatmap(T)

current_state_zero = np.argmax(
    X_test_zero[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')]].values, 
    axis=1
)
T_zero = transition_matrix(current_state_zero, y_pred_proba_zero)
print("Transition Matrix for for previous state 0:\n", T_zero)
plot_transition_heatmap(T_zero)

current_state_nonzero = np.argmax(
    X_test_nonzero[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')]].values, 
    axis=1
)
T_nonzero = transition_matrix(current_state_nonzero, y_pred_proba_nonzero)
print("Transition Matrix for for previous state 1:\n", T_nonzero)
plot_transition_heatmap(T_nonzero)

#################################### Evaluations ###################################

average_probability = mean_prob(y_pred_proba, y_test.to_numpy())
print(f'average probability = {average_probability}\n')
average_probability_zero = mean_prob(y_pred_proba_zero, y_test_zero.to_numpy())
print(f'average probability for previous state 0 = {average_probability_zero}\n')
average_probability_nonzero = mean_prob(y_pred_proba_nonzero, y_test_nonzero.to_numpy())
print(f'average probability for previous state 1 = {average_probability_nonzero}\n')

entropy_probability = entropy_sample(y_pred_proba, y_test.to_numpy())
print(f'entropy_sample = {entropy_probability}\n')
entropy_probability_zero = entropy_sample(y_pred_proba_zero, y_test_zero.to_numpy())
print(f'entropy_sample for previous state 0 = {average_probability_zero}\n')
entropy_probability_nonzero = entropy_sample(y_pred_proba_nonzero, y_test_nonzero.to_numpy())
print(f'entropy_sample for previous state 1 = {entropy_probability_nonzero}\n')

entropy_class_probability = entropy_class(y_pred_proba, y_test.to_numpy())
print(f'entropy_class = {entropy_class_probability}\n')
entropy_class_probability_zero = entropy_class(y_pred_proba_zero, y_test_zero.to_numpy())
print(f'entropy_class for previous state 0 = {entropy_class_probability_zero}\n')
entropy_class_probability_nonzero = entropy_class(y_pred_proba_nonzero, y_test_nonzero.to_numpy())
print(f'entropy_class for previous state 1 = {entropy_class_probability_nonzero}\n')

brier_score = brier(y_pred_proba, y_test.to_numpy())
print(f'brier score = {brier_score}\n')
brier_score_zero = brier(y_pred_proba_zero, y_test_zero.to_numpy())
print(f'brier score for previous state 0 = {brier_score_zero}\n')
brier_score_nonzero = brier(y_pred_proba_nonzero, y_test_nonzero.to_numpy())
print(f'brier score for previous state 1 = {brier_score_nonzero}\n')

brier_score = brier_weighted(y_pred_proba, y_test.to_numpy())
print(f'adjusted brier score = {brier_score}\n')
brier_score_zero = brier_weighted(y_pred_proba_zero, y_test_zero.to_numpy())
print(f'adjusted brier score for previous state 0 = {brier_score_zero}\n')
brier_score_nonzero = brier_weighted(y_pred_proba_nonzero, y_test_nonzero.to_numpy())
print(f'adjusted brier score for previous state 1 = {brier_score_nonzero}\n')
