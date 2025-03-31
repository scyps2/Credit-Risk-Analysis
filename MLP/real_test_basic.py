import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('2007Q1_nonstandard_performance.csv').head(10000)

df = df[[
    "Loan Sequence Number", "Monthly Reporting Period", 
    "Current Loan Delinquency Status"
]]

# data preprocess, creat states of next month and clean dataset
def preprocess(df):
    df = df.sort_values(by=['Loan Sequence Number', "Monthly Reporting Period"])
    df['Next Loan Delinquency Status'] = df.groupby('Loan Sequence Number')['Current Loan Delinquency Status'].shift(-1)
    df = df.dropna()
    df = df[df["Current Loan Delinquency Status"] != "RA"]
    df = df[df["Next Loan Delinquency Status"] != "RA"]
    df = df.reset_index(drop=True)
    return df

df = preprocess(df)
print(df.head(), df.shape)

# Encode y and y_next to one hot form
inputs = ["Current Loan Delinquency Status", 'Next Loan Delinquency Status']
def one_hot_encoder(df):
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[inputs])
    df_one_hot = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(inputs))
    df = pd.concat([df, df_one_hot], axis=1)
    return df, encoder

df, encoder = one_hot_encoder(df)
encoded_columns = encoder.get_feature_names_out(inputs)

X = df[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')]]
y = df[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MLP Classifying
mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 5000, random_state = 1,
                   learning_rate_init = 0.0001, learning_rate = 'adaptive')

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
y_pred_proba = mlp.predict_proba(X_test)

# Evaluation by brier score
def brier(y_pred_proba, y_test):
    score_matrix = (y_pred_proba - y_test)**2
    brier_score_states = np.mean(score_matrix, axis = 0)
    for i, score in enumerate(brier_score_states):
        print(f"Brier score for state {i} is {score}")
    brier_score = np.sum(brier_score_states)
    return brier_score

brier_score = brier(y_pred_proba, y_test)
print('brier score = ', brier_score)

accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)