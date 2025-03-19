import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

df_test = pd.read_csv('data/simCRdata_test3.csv')
df_train = pd.read_csv('data/simCRdata_train3.csv')

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

brier_scores = []
for i in range (0, 10):
    # search for the best parameter set randomly
    mlp = MLPClassifier()
    param_dist = {
        'hidden_layer_sizes': [(randint.rvs(10, 50), randint.rvs(10, 50)) for _ in range(10)],
        'activation': ['relu', 'tanh', 'logistic'],
        'max_iter' : [500],
        'learning_rate_init': np.linspace(0.0001, 1, 100),
        'learning_rate': ['constant', 'adaptive']
    }

    def brier(y_pred_proba, y_test):
        score_matrix = (y_pred_proba - y_test)**2
        brier_score_states = np.mean(score_matrix, axis = 0)
        brier_score = np.sum(brier_score_states)
        return brier_score

    scorer = make_scorer(brier, greater_is_better=False)
    search = RandomizedSearchCV(estimator=mlp, param_distributions=param_dist, cv=3, scoring=scorer, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best parameters: {search.best_params_}")

    best_mlp = search.best_estimator_
    y_pred = best_mlp.predict(X_test)
    y_pred_proba = best_mlp.predict_proba(X_test)

    # Evaluation by brier score
    brier_score = brier(y_pred_proba, y_test)
    # print(f"brier score = {brier_score}")
    print(f"brier score for iter {i} = {brier_score}")
    brier_scores.append(brier_score)

accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

average_brier_score = np.mean(brier_scores)
print(f"brier score = {average_brier_score}")

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