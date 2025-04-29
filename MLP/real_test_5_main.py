import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

MONTH_AHEAD = 3

df_performance = pd.read_csv('2020Q1_standard_performance.csv').head(500000)
df_origination = pd.read_csv('2020Q1_standard_origination.csv').head(500000)

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
    threshold = 0.995
    n = int(status_summary[status_summary["Cumulative Ratio"] <= threshold]["Delinquency Status"].max())
    # df["Processed Loan Delinquency Status"] = df["Current Loan Delinquency Status"].apply(
    #     lambda x: x if x <= n else f"{n}+"
    # )
    # df['Current Loan Delinquency Status'] = df['Processed Loan Delinquency Status'].astype(str)
    df["Current Loan Delinquency Status"] = df["Current Loan Delinquency Status"].apply(
        lambda x: x if x <= n else n+1
    )
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

X_train = train_df[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')] +
        ["Credit Score", "Original UPB", "Original Debt-to-Income (DTI) Ratio", 
        "Original Loan-to-Value (LTV)", "Original Interest Rate"]]
y_train = train_df[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]

X_test = test_df[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')] +
        ["Credit Score", "Original UPB", "Original Debt-to-Income (DTI) Ratio", 
        "Original Loan-to-Value (LTV)", "Original Interest Rate"]]
y_test = test_df[[col for col in encoded_columns if col.startswith('Next Loan Delinquency Status')]]

# MLP Classifying
mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 5000, random_state = 1,
                   learning_rate_init = 0.0001, learning_rate = 'adaptive')

mlp.fit(X_train, y_train)

def predict_n_months(mlp, n, input):
    features = input[[
        "Credit Score", "Original UPB", "Original Debt-to-Income (DTI) Ratio", 
        "Original Loan-to-Value (LTV)", "Original Interest Rate"
        ]].to_numpy()
    for _ in range(n):
        pred_proba = mlp.predict_proba(input)

        # remain probability form
        # input = pred_proba
        # encode to one hot
        input_labels = np.argmax(pred_proba, axis=1)
        input = np.zeros_like(pred_proba)
        input[np.arange(len(input)), input_labels] = 1

        input = np.hstack((input, features))

    return pred_proba

y_pred_proba = predict_n_months(mlp, MONTH_AHEAD, X_test)
# y_pred_proba = mlp.predict_proba(X_test)

# generate transition matrix and visualization
def transition_matrix(current_state, y_pred_proba):
    num_classes = y_pred_proba.shape[1]
    transition_matrix = np.zeros((num_classes, num_classes))

    for row in range(len(current_state)): # iterate over rows
        from_state = current_state[row]
        transition_matrix[from_state] += y_pred_proba[row]

    row_sum = transition_matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1 # state not appear in test set
    transition_matrix = transition_matrix / row_sum
    
    return transition_matrix

def plot_transition_heatmap(transition_matrix):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        transition_matrix, 
        fmt=".2f", cmap="Blues"
    )
    ax.xaxis.set_ticks_position('top')

    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.title("Transition Matrix Heatmap")
    plt.tight_layout()
    plt.show()

current_state = np.argmax( # decode into integer columns(before one-hot)
    X_test[[col for col in encoded_columns if col.startswith('Current Loan Delinquency Status')]].values,
    axis=1
)
T = transition_matrix(current_state, y_pred_proba)
print("Transition Matrix:\n", T)
plot_transition_heatmap(T)

# Evaluation by mean probability
# Range: (0, 1)
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

# Evaluation by log probability with base `e`
# Range: (0, ln(num_classes))
def entropy(y_pred_proba, y_test):
    true_probs = np.sum(y_pred_proba * y_test, axis=1)
    log_probs = np.empty_like(true_probs)
    for i, p in enumerate(true_probs):
        if p == 0:
            print(f"Sample {i}: True class probability is 0, setting log to NaN")
            log_probs[i] = np.nan
        else:
            log_probs[i] = np.log(p)

    entropy = -np.nanmean(log_probs)
    return entropy

average_probability = mean_prob(y_pred_proba, y_test.to_numpy())
print(f'average probability = {average_probability}')

entropy_probability = entropy(y_pred_proba, y_test.to_numpy())
print(f'\nentropy = {entropy_probability}')

# Evaluation by brier score
def brier(y_pred_proba, y_test):
    score_matrix = (y_pred_proba - y_test)**2
    brier_score_states = np.mean(score_matrix, axis = 0)
    for i, score in enumerate(brier_score_states):
        print(f"Brier score for state {i} is {score}")
    brier_score = np.sum(brier_score_states)
    return brier_score

def brier_weighted(y_pred_proba, y_test, distance_power = 1):
    score_matrix = (y_pred_proba - y_test)**2

    # decode one hot
    true_labels = np.argmax(y_test, axis=1)
    num_classes = y_pred_proba.shape[1]

    weighted_scores = []

    for i, true_label in enumerate(true_labels):
        # calculate weight list
        distances = np.abs(np.arange(num_classes) - true_label)
        weights = (distances + 1) ** distance_power

        weighted_score = weights * score_matrix[i] / np.sum(weights)
        weighted_scores.append(weighted_score)

    brier_score_states = np.mean(weighted_scores, axis = 0)
    for i, score in enumerate(brier_score_states):
        print(f"Brier score for state {i} is {score}")
    brier_score = np.sum(brier_score_states)

    return brier_score

print("\nOverall brier score")
brier_score = brier(y_pred_proba, y_test.to_numpy())
print('brier score = ', brier_score)
brier_score = brier_weighted(y_pred_proba, y_test.to_numpy())
print('adjusted brier score = ', brier_score)