import pandas as pd
from sklearn.neural_network import MLPClassifier

from utils import *
from evaluations import *

################################### configurations ##############################

DATA_AMOUNT = 1000000

### MLP + credit + 4features + time
features_performance = ["Loan Age"]
features_origination = ["Credit Score", "Original UPB", "Original Debt-to-Income (DTI) Ratio", 
        "Original Loan-to-Value (LTV)", "Original Interest Rate"]

### MLP + credit
# features_performance = []
# features_origination = ["Credit Score"]

### MLP
# features_performance = []
# features_origination = []

features = features_performance + features_origination + ['overall_rate', 'USA (SA)']
MONTH_AHEAD = 6
predictor = Predictor()

################################# processing ####################################

df_performance = pd.read_csv('useful_data/2020Q1_standard_performance.csv').head(DATA_AMOUNT)
df_origination = pd.read_csv('useful_data/2020Q1_standard_origination.csv')
df_unemployment = pd.read_csv('useful_data/unemployment_rates.csv')
df_hpi = pd.read_csv('useful_data/hpi_po_monthly_hist.csv')

df_performance = df_performance[["Loan Sequence Number", "Monthly Reporting Period", 
    "Current Loan Delinquency Status"] + features_performance]
df_origination = df_origination[["Loan Sequence Number"] + features_origination]
df = pd.merge(df_origination, df_performance, on="Loan Sequence Number", how="inner")

df_unemployment = df_unemployment[['date', 'overall_rate']]
df_unemployment['date'] = pd.to_datetime(df_unemployment['date']).dt.strftime('%Y%m').astype(int)
df = pd.merge(df_unemployment, df, left_on='date', right_on='Monthly Reporting Period', how="inner")
df = df.drop(columns=['date'])

df_hpi = df_hpi[['Month', 'USA (SA)']]
df_hpi['Month'] = pd.to_datetime(df_hpi['Month']).dt.strftime('%Y%m').astype(int)
df = pd.merge(df_hpi, df, left_on='Month', right_on='Monthly Reporting Period', how="inner")
df = df.drop(columns=['Month'])

print(df.head())

df = reclassify(df)
df = preprocess(df)
print(df.head())

columns_to_encode = ["Current Loan Delinquency Status", "Next Loan Delinquency Status"]
df = one_hot_encoder(df, columns_to_encode)
Xys = train_test_split_loan(df, features)

# MLP prediction
mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 5000, random_state = 1,
                   learning_rate_init = 0.0001, learning_rate = 'adaptive')

for month in range(1, MONTH_AHEAD+1):
    print(f"----------------month {month}----------------")
    _, prediction_dict = fit_and_predict(mlp, Xys, predictor, "weighted", month, features)
    # calculate_transition_matrix(prediction_dict, Xys, df)

    y_pred_proba = prediction_dict["y_pred_proba"]
    y_test = Xys["y_test"]
    entropy = entropy_sample(y_pred_proba, y_test.to_numpy())
    print(f'entropy_sample = {entropy}')
    class_entropy = entropy_class(y_pred_proba, y_test.to_numpy())
    print(f'entropy_class = {class_entropy}\n')

    # y_pred_proba_nonzero = prediction_dict["y_pred_proba_nonzero"]
    # y_test_nonzero = Xys["y_test_nonzero"]
    # entropy_nonzero = entropy_sample(y_pred_proba_nonzero, y_test_nonzero.to_numpy())
    # print(f'entropy_sample for initial deliquencies = {entropy_nonzero}')
    # class_entropy_nonzero = entropy_class(y_pred_proba_nonzero, y_test_nonzero.to_numpy())
    # print(f'entropy_class for initial deliquencies = {class_entropy_nonzero}')
