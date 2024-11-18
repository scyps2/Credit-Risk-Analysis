import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

df_test = pd.read_csv('data/simCRdata_test5.csv')
df_train = pd.read_csv('data/simCRdata_train5.csv')

# creat states of next month and clean dataset, rescale mev
def preprocess(df):
    df = df.sort_values(by=['cust', 't'])
    df['y_next'] = df.groupby('cust')['y'].shift(-1)
    df = df.dropna()
    df['y_next'] = df['y_next'].astype(int)

    # standarization
    # mean_mev = df['mev'].mean()
    # std_mev = df['mev'].std()
    # df['mev'] = (df['mev'] - mean_mev) / std_mev

    # normalization
    # mev_min = np.min(df['mev'])
    # mev_max = np.max(df['mev'])
    # df['mev'] = (df['mev'] - mev_min) / (mev_max - mev_min)

    return df

df_test = preprocess(df_test)
df_train = preprocess(df_train)

# seperate by grade
def seperate_grade(df):
    df_grade_0 = preprocess(df[df['grade'] == 0])
    df_grade_1 = preprocess(df[df['grade'] == 1])
    return df_grade_0, df_grade_1

df_train_0, df_train_1 = seperate_grade(df_train)
df_test_0, df_test_1 = seperate_grade(df_test)

# Encode y and y_next to one hot form
inputs = ['y', 'y_next', 'grade', 'mev']
def one_hot_encoder(df):
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[inputs])
    df_one_hot = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(inputs))
    df = pd.concat([df, df_one_hot], axis=1)
    return df

df_train_0 = one_hot_encoder(df_train_0)
df_train_1 = one_hot_encoder(df_train_1)
df_test_0 = one_hot_encoder(df_test_0)
df_test_1 = one_hot_encoder(df_test_1)
print(df_train_0.head())

# MLP Classifying
X_train_0 = df_train_0[['y_0', 'y_1', 'y_2', 'y_3', 'grade', 'mev_-1', 'mev_1']].dropna().to_numpy()
y_train_0 = df_train_0[['y_next_0', 'y_next_1', 'y_next_2', 'y_next_3']].dropna().to_numpy()
X_test_0 = df_test_0[['y_0', 'y_1', 'y_2', 'y_3', 'grade', 'mev_-1', 'mev_1']].dropna().to_numpy()
y_test_0 = df_test_0[['y_next_0', 'y_next_1', 'y_next_2', 'y_next_3']].dropna().to_numpy()

X_train_1 = df_train_1[['y_0', 'y_1', 'y_2', 'y_3', 'grade', 'mev_-1', 'mev_1']].dropna().to_numpy()
y_train_1 = df_train_1[['y_next_0', 'y_next_1', 'y_next_2', 'y_next_3']].dropna().to_numpy()
X_test_1 = df_test_1[['y_0', 'y_1', 'y_2', 'y_3', 'grade', 'mev_-1', 'mev_1']].dropna().to_numpy()
y_test_1 = df_test_1[['y_next_0', 'y_next_1', 'y_next_2', 'y_next_3']].dropna().to_numpy()

shared_rows = min(len(X_train_0), len(y_train_0))
X_train_0 = X_train_0[:shared_rows]
y_train_0 = y_train_0[:shared_rows]
X_test_0 = X_test_0[:shared_rows]
y_test_0 = y_test_0[:shared_rows]

shared_rows = min(len(X_train_1), len(y_train_1))
X_train_1 = X_train_1[:shared_rows]
y_train_1 = y_train_1[:shared_rows]
X_test_1 = X_test_1[:shared_rows]
y_test_1 = y_test_1[:shared_rows]

mlp = MLPClassifier(hidden_layer_sizes = (8, ), activation = 'relu', max_iter = 5000, random_state = 1,
                   learning_rate_init = 0.0001, learning_rate = 'adaptive')

mlp.fit(X_train_0, y_train_0)
y_pred_0 = mlp.predict(X_test_0)
y_pred_proba_0 = mlp.predict_proba(X_test_0)
# print(f"Probability matrix for grade 0 is: {y_pred_proba_0}")

mlp.fit(X_train_1, y_train_1)
y_pred_1 = mlp.predict(X_test_1)
y_pred_proba_1 = mlp.predict_proba(X_test_1)
# print(f"Probability matrix for grade 1 is: {y_pred_proba_1}")

# Evaluation by brier score
def brier(y_pred_proba, y_test):
    score_matrix = (y_pred_proba - y_test)**2
    brier_score_states = np.mean(score_matrix, axis = 0)
    brier_score = np.sum(brier_score_states)
    return brier_score

brier_score_0 = brier(y_pred_proba_0, y_test_0)
print('brier score for grade 0 = ', brier_score_0)
brier_score_1 = brier(y_pred_proba_1, y_test_1)
print('brier score for grade 1 = ', brier_score_1)

# ROC curve
# fig, axs = plt.subplots(2, 2)
# for i, ax in enumerate(axs.ravel()):
#     fpr, tpr, thresholds = roc_curve(y_test[:, i], y_pred_proba[:, i])
#     roc_auc = auc(fpr, tpr)
#     ax.plot(fpr, tpr, color = 'blue', label = 'area = %0.4f' % roc_auc)
#     ax.legend(loc = "lower right")
#     ax.set_xlabel('False Positive Rate')
#     ax.set_ylabel('True Positive Rate')
#     ax.set_title(f'ROC Curve of state {i}')

#     # find the point which is nearest to TPR = 1
#     threshold_index = np.argmin(np.abs(tpr - 1))
#     ax.scatter(fpr[threshold_index], tpr[threshold_index])
#     print(f"Best threshold for state {i} is {thresholds[threshold_index]: .2f}")

# plt.tight_layout()
# plt.show()