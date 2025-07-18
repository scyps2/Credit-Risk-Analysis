import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df_performance = pd.read_csv('2020Q1_standard_performance.csv').head(100000)
df_origination = pd.read_csv('2020Q1_standard_origination.csv').head(100000)

df_performance = df_performance[[
    "Loan Sequence Number", "Monthly Reporting Period", 
    "Current Loan Delinquency Status"
]]
df_origination = df_origination[[
    "Loan Sequence Number", "Credit Score", "Original UPB", 
    "Original Debt-to-Income (DTI) Ratio", "Original Loan-to-Value (LTV)",
    "Original Interest Rate"
]]
df = pd.merge(df_origination, df_performance, on="Loan Sequence Number", how="inner")
print(df.head(), df.shape)


def reclassify(df):
    df = df.dropna()
    df = df[df["Current Loan Delinquency Status"] != "RA"]

    # frequency summary for Current Loan Delinquency Status
    df['Current Loan Delinquency Status'] = df['Current Loan Delinquency Status'].astype(int)
    status_counts = df["Current Loan Delinquency Status"].value_counts().sort_index()
    total_count = len(df)
    cumulative_ratio = status_counts.cumsum() / total_count

    status_summary = pd.DataFrame({
        "Delinquency Status": status_counts.index,
        "Frequency": status_counts.values,
        "Cumulative Ratio": cumulative_ratio.values
    }).sort_values(by="Delinquency Status", ascending=True)

    print("Delinquency Status | Frequency | Cumulative Ratio")
    print("-" * 50)
    for _, row in status_summary.iterrows():
        print(f"{row['Delinquency Status']:<17} | {row['Frequency']:<9} | {row['Cumulative Ratio']:.4f}")

    # reclassify
    threshold = 0.99
    n = int(status_summary[status_summary["Cumulative Ratio"] <= threshold]["Delinquency Status"].max())
    df["Processed Loan Delinquency Status"] = df["Current Loan Delinquency Status"].apply(
        lambda x: x if x <= n else f"{n}+"
    )
    df['Current Loan Delinquency Status'] = df['Processed Loan Delinquency Status'].astype(str)
    print(df["Current Loan Delinquency Status"].unique())

    return df

df = reclassify(df)

def preprocess(df):
    # creat states of next month
    df = df.sort_values(by=['Loan Sequence Number', "Monthly Reporting Period"])
    df['Next Loan Delinquency Status'] = df.groupby('Loan Sequence Number')['Current Loan Delinquency Status'].shift(-1)
    df = df.dropna()
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

# X = df[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')]]
X = df[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')] +
        ["Credit Score", "Original UPB", "Original Debt-to-Income (DTI) Ratio", 
        "Original Loan-to-Value (LTV)", "Original Interest Rate"]]
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
    brier_score = np.sum(brier_score_states) # mean or sum?
    return brier_score

brier_score = brier(y_pred_proba, y_test)
print('brier score = ', brier_score)

accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)