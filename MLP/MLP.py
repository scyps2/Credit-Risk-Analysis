import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

df_test = pd.read_csv('simCRdata_test.csv')
############## test for this data 
df_train = pd.read_csv('simCRdata.csv')

# data preprocess, creat states of next month and clean dataset
def preprocess(df):
    df = df.sort_values(by=['cust', 't'])
    df['y_next'] = df.groupby('cust')['y'].shift(-1)
    df = df.dropna()
    df['y_next'] = df['y_next'].astype(int)
    return df

df_test = preprocess(df_test)
df_train = preprocess(df_train)

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
print(df_train.head())

# MLP Classifying
X_train = df_train[['y_0', 'y_1', 'y_2', 'y_3', 'grade_0', 'grade_1']].dropna().to_numpy()
y_train = df_train[['y_next_0', 'y_next_1', 'y_next_2', 'y_next_3']].dropna().to_numpy()
X_test = df_test[['y_0', 'y_1', 'y_2', 'y_3', 'grade_0', 'grade_1']].dropna().to_numpy()
y_test = df_test[['y_next_0', 'y_next_1', 'y_next_2', 'y_next_3']].dropna().to_numpy()

mlp = MLPClassifier(hidden_layer_sizes = (10, 10), activation = 'relu', max_iter = 500, random_state = 1,
                   learning_rate_init = 0.01, learning_rate = 'adaptive')

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
y_pred_proba = mlp.predict_proba(X_test)
# print(y_pred_proba)

# Evaluation by brier score
def brier_score(predicted, actual):
    return np.mean(np.sum((predicted - actual)**2, axis = 1))

average_brier_score = brier_score(y_pred_proba, y_test)
print('Average brier score by probability is', average_brier_score)

accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# ROC curve
fig, axs = plt.subplots(2, 2)
for i, ax in enumerate(axs.ravel()):
    fpr, tpr, thresholds = roc_curve(y_test[:, i], y_pred_proba[:, i])
    ax.plot(fpr, tpr, color = 'blue')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve of state {i}')

    # find the point which is nearest to TPR = 1
    threshold_index = np.argmin(np.abs(tpr - 1))
    ax.scatter(fpr[threshold_index], tpr[threshold_index])
    print(f"Best threshold for state {i} is {thresholds[threshold_index]: .2f}")

plt.tight_layout()
plt.show()