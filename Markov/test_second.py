import numpy as np
import pandas as pd

transition_matrix = np.array([
    [0.961749,0.038251,0.000000,0.000000],
    [0.011111,0.177778,0.211111,0.600000],
    [0.000000,0.602740,0.068493,0.328767],
    [0.000000,0.009524,0.542857,0.447619]
    ])
Identity_matrix = np.eye(4)
df_test = pd.read_csv('simCRdata_test.csv')
df_test = df_test.sort_values(by = ['cust','t'])
print(df_test.head())

print('Here is the transition matrix deriverd from the training data')
print(transition_matrix)

df_test['y_next'] = df_test.groupby('cust')['y'].shift(-1)
df_clean = df_test.dropna(subset = ['y_next'])
#print(df_clean.head())

# MSE
def brier_score_multiclass(predicted_probs, actual):
    actual_one_hot = np.zeros_like(predicted_probs)
    actual_one_hot[actual] = 1
    return np.sum((predicted_probs - actual_one_hot)**2)

brier_scores=[]
Identity_brier_scores=[]

for index,row in df_clean.iterrows():
    current_state = int(row['y'])
    next_state = int(row['y_next'])

    predicted_probs = transition_matrix[current_state]
    Identity_predicted_probs = Identity_matrix[current_state]

    score = brier_score_multiclass(predicted_probs, next_state)
    Identity_score = brier_score_multiclass(Identity_predicted_probs, next_state)
    brier_scores.append(score)
    Identity_brier_scores.append(Identity_score)

df_clean['brier_scores'] = brier_scores
df_clean['brier_scores of Identity Matrix'] = Identity_brier_scores
df_clean.to_csv('test_data.csv',index = False)
print(df_clean.head())

average_brier_score = np.mean(brier_scores)
average_brier_score_identity = np.mean(Identity_brier_scores)
print('average brier score is')
print(average_brier_score)
print('average brier score when transition matrix is Identity matrix is')
print(average_brier_score_identity)