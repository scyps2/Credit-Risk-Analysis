import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from evaluations import *

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

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

    # print("Delinquency Status | Frequency | Cumulative Ratio")
    # print("-" * 50)
    # for _, row in status_summary.iterrows():
    #     print(f"{row['Delinquency Status']:<17} | {row['Frequency']:<9} | {row['Cumulative Ratio']:.4f}")

    # reclassify
    threshold = 0.995
    months_ahead = int(status_summary[status_summary["Cumulative Ratio"] <= threshold]["Delinquency Status"].max())
    # df["Processed Loan Delinquency Status"] = df["Current Loan Delinquency Status"].apply(
    #     lambda x: x if x <= months_ahead else f"{months_ahead}+"
    # )
    # df['Current Loan Delinquency Status'] = df['Processed Loan Delinquency Status'].astype(str)
    df["Current Loan Delinquency Status"] = df["Current Loan Delinquency Status"].apply(
        lambda x: x if x <= 6 else 7
    )
    # print(df["Current Loan Delinquency Status"].unique())
    return df

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

def one_hot_encoder(df, columns_to_encode):
    # Encode y and y_next to one hot form
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[columns_to_encode])
    df_one_hot = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(columns_to_encode))
    df = df.drop(columns=columns_to_encode)
    df = pd.concat([df, df_one_hot], axis=1)
    return df

def train_test_split_loan(df, features=[]):
    # split dataset according to loans
    unique_loans = df["Loan Sequence Number"].unique()
    train_loans, test_loans = train_test_split(unique_loans, test_size=0.3, random_state=42)
    train_df = df[df["Loan Sequence Number"].isin(train_loans)]
    test_df = df[df["Loan Sequence Number"].isin(test_loans)]

    X_candidate_cols = [col for col in df.columns if col.startswith('Current Loan Delinquency Status')] + features
    y_candidate_cols = [col for col in df.columns if col.startswith('Next Loan Delinquency Status')]
    
    X_train = train_df[[col for col in X_candidate_cols if col in train_df.columns]]
    y_train = train_df[[col for col in y_candidate_cols if col in train_df.columns]]
    X_test = test_df[[col for col in X_candidate_cols if col in test_df.columns]]
    y_test = test_df[[col for col in y_candidate_cols if col in test_df.columns]]

    # Optional: separate dataset by previous deliquency state
    train_zero = train_df[train_df["Current Loan Delinquency Status_0"] == 1]
    test_zero = test_df[test_df["Current Loan Delinquency Status_0"] == 1]
    X_train_zero = train_zero[[col for col in X_candidate_cols if col in train_df.columns]]
    y_train_zero = train_zero[[col for col in y_candidate_cols if col in train_df.columns]]                                                                    
    X_test_zero = test_zero[[col for col in X_candidate_cols if col in test_df.columns]]
    y_test_zero = test_zero[[col for col in y_candidate_cols if col in test_df.columns]]

    train_nonzero = train_df[train_df["Current Loan Delinquency Status_0"] == 0]
    test_nonzero = test_df[test_df["Current Loan Delinquency Status_0"] == 0]
    X_train_nonzero = train_nonzero[[col for col in X_candidate_cols if col in train_df.columns]]
    y_train_nonzero = train_nonzero[[col for col in y_candidate_cols if col in train_df.columns]]
    X_test_nonzero = test_nonzero[[col for col in X_candidate_cols if col in test_df.columns]]
    y_test_nonzero = test_nonzero[[col for col in y_candidate_cols if col in test_df.columns]]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_train_zero": X_train_zero, "y_train_zero": y_train_zero,
        "X_test_zero": X_test_zero, "y_test_zero": y_test_zero,
        "X_train_nonzero": X_train_nonzero, "y_train_nonzero": y_train_nonzero,
        "X_test_nonzero": X_test_nonzero, "y_test_nonzero": y_test_nonzero
    }

class Predictor:
    @staticmethod
    def basic(mlp, months_ahead, feature_columns, input):
        features = input[feature_columns].to_numpy()
        status_cols = [col for col in input.columns if col.startswith("Current Loan Delinquency Status")]
        current = input[status_cols].to_numpy()
        for month in range(months_ahead):
            pred_proba = mlp.predict_proba(np.hstack((current, features)))
            current = pred_proba
        return pred_proba

    @staticmethod
    def weighted(mlp, months_ahead, feature_columns, input):
        # current distribution in one-hot form
        current_states = input[[col for col in input.columns if col.startswith('Current Loan Delinquency Status')]].to_numpy()
        num_classes = current_states.shape[1]
        results = None

        features = input[feature_columns].to_numpy()
        for _ in range(months_ahead):
            # predict next month probability distributions
            next_states = mlp.predict_proba(np.hstack((current_states, features)))

            # assume next state is 0,1...6+
            predicted_given_each_next_state = []
            for assumed_next_state in range(num_classes):
                one_hot_next_state = np.zeros_like(next_states)
                one_hot_next_state[:, assumed_next_state] = 1
                predicted_probs = mlp.predict_proba(np.hstack((one_hot_next_state, features)))
                predicted_given_each_next_state.append(predicted_probs)

            # change list of arrays to 3d array
            predicted_given_each_next_state = np.stack(predicted_given_each_next_state, axis=1) 
            # weight by transition probabilities
            results = np.einsum('bi,bij->bj', next_states, predicted_given_each_next_state)
            current_states = results

        return results
    
def fit_and_predict(mlp, Xys, predictor: Predictor, strategy, months_ahead, feature_columns):
    mlps = {}
    predictions = {}

    for subset in ["", "_nonzero"]:
        X_train = Xys[f"X_train{subset}"]
        y_train = Xys[f"y_train{subset}"]
        X_test  = Xys[f"X_test{subset}"]

        y_train = np.argmax(y_train, axis=1)
        # print(f"labels contain: {np.unique(y_train)}")
        mlp.fit(X_train, y_train)

        if strategy == "basic":
            y_pred_proba = predictor.basic(mlp, months_ahead, feature_columns, X_test)
        elif strategy == "weighted":
            y_pred_proba = predictor.weighted(mlp, months_ahead, feature_columns, X_test)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        mlps[subset] = mlp
        predictions[f"y_pred_proba{subset}"] = y_pred_proba

    return mlps, predictions

def calculate_transition_matrix(predictions: dict, Xys, df, prefix="y_pred_proba"):
    matrices = {}
    for subset in ["", "_nonzero"]:
        y_pred = predictions[f"{prefix}{subset}"]

        # decode into integer columns(before one-hot)
        X_test = Xys[f"X_test{subset}"]
        current_states = np.argmax(
            X_test[[col for col in df.columns if col.startswith("Current Loan Delinquency Status")]].values,
            axis=1
        )
        T = transition_matrix(current_states, y_pred)
        matrices[subset] = T

        pd.DataFrame(T).to_csv(f"transition_matrix{subset}.csv", index=False)
        plot_transition_heatmap(T)