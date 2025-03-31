import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('2020Q1_standard_performance.csv').head(100000)

df = df[[
    "Loan Sequence Number", "Monthly Reporting Period", 
    "Current Loan Delinquency Status"
]]

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
    threshold = 0.995
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

    # standarization
    exclude_cols = ['Loan Sequence Number', 'Monthly Reporting Period', 
                    'Current Loan Delinquency Status', 'Next Loan Delinquency Status']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_standardize = [col for col in numeric_cols if col not in exclude_cols]
    for col in cols_to_standardize:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std

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

# split dataset according to loans
unique_loans = df["Loan Sequence Number"].unique()
train_loans, test_loans = train_test_split(unique_loans, test_size=0.3, random_state=42)
train_df = df[df["Loan Sequence Number"].isin(train_loans)]
test_df = df[df["Loan Sequence Number"].isin(test_loans)]

X_train = train_df[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')]]
y_train = train_df[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]
X_test = test_df[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')]]
y_test = test_df[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]

# MLP Classifying
mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 5000, random_state = 1,
                   learning_rate_init = 0.0001, learning_rate = 'adaptive')

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
y_pred_proba = mlp.predict_proba(X_test)

# y_pred = mlp.predict(X_train)
# y_pred_proba = mlp.predict_proba(X_train)

# Evaluation by mean probability
def mean_prob(y_pred_proba, y_test):
    mean_prob_class = []
    for i in range(y_pred_proba.shape[1]):
        rows_i = y_test[:, i] == 1 # select all rows whose true label is i (boolean)
        if np.sum(rows_i) > 0:
            mean_prob_i = np.mean(y_pred_proba[rows_i, i]) # y_pred_proba[rows_i, i]: only calculate rows of True
        else:
            mean_prob_i = np.nan
        print(f"Probability of truly predicting class {i} is {mean_prob_i}")
        mean_prob_class.append(mean_prob_i)

    mean_prob = np.nanmean(mean_prob_class)
    return mean_prob

average_probability = mean_prob(y_pred_proba, y_test.to_numpy())
# average_probability = mean_prob(y_pred_proba, y_train.to_numpy())
print(f'average probability = {average_probability}')

# Evaluation by brier score
def brier(y_pred_proba, y_test):
    score_matrix = (y_pred_proba - y_test)**2
    brier_score_states = np.mean(score_matrix, axis = 0)
    for i, score in enumerate(brier_score_states):
        print(f"Brier score for state {i} is {score}")
    brier_score = np.sum(brier_score_states)
    return brier_score

brier_score = brier(y_pred_proba, y_test)
# brier_score = brier(y_pred_proba, y_train)
print('brier score = ', brier_score)