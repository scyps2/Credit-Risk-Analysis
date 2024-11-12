import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

df_test = pd.read_csv('data/simCRdata_test4.csv')
df_train = pd.read_csv('data/simCRdata_train4.csv')

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
    mev_min = np.min(df['mev'])
    mev_max = np.max(df['mev'])
    df['mev'] = (df['mev'] - mev_min) / (mev_max - mev_min)

    return df

df_test = preprocess(df_test)
df_train = preprocess(df_train)
print(df_train.head())

# Encode y and y_next to one hot form
inputs = ['y', 'y_next', 'grade']
def one_hot_encoder(df):
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[inputs])
    df_one_hot = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(inputs))
    df = pd.concat([df, df_one_hot], axis=1)
    return df

df_test = one_hot_encoder(df_test)
df_train = one_hot_encoder(df_train)

# MLP Classifying
X_train = df_train[['y_0', 'y_1', 'y_2', 'y_3', 'grade_0', 'grade_1', 'mev']].dropna().to_numpy()
y_train = df_train[['y_next_0', 'y_next_1', 'y_next_2', 'y_next_3']].dropna().to_numpy()
X_test = df_test[['y_0', 'y_1', 'y_2', 'y_3', 'grade_0', 'grade_1', 'mev']].dropna().to_numpy()
y_test = df_test[['y_next_0', 'y_next_1', 'y_next_2', 'y_next_3']].dropna().to_numpy()

shared_rows = min(len(X_train), len(y_train))
X_train = X_train[:shared_rows]
y_train = y_train[:shared_rows]
X_test = X_test[:shared_rows]
y_test = y_test[:shared_rows]

mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 500, random_state = 1,
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

# ROC curve
fig, axs = plt.subplots(2, 2)
for i, ax in enumerate(axs.ravel()):
    fpr, tpr, thresholds = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color = 'blue', label = 'area = %0.4f' % roc_auc)
    ax.legend(loc = "lower right")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve of state {i}')

    # find the point which is nearest to TPR = 1
    threshold_index = np.argmin(np.abs(tpr - 1))
    ax.scatter(fpr[threshold_index], tpr[threshold_index])
    print(f"Best threshold for state {i} is {thresholds[threshold_index]: .2f}")

plt.tight_layout()
plt.show()