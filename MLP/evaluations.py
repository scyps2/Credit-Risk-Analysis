import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def transition_matrix(current_states, y_pred_proba):
    num_classes = y_pred_proba.shape[1]
    transition_matrix = np.zeros((num_classes, num_classes))

    for row in range(len(current_states)): # iterate over rows
        from_state = current_states[row]
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

# Evaluation by mean probability
def mean_prob(y_pred_proba, y_test):
    mean_prob_class = []
    # iterate over classes
    for i in range(y_pred_proba.shape[1]):
        rows_i = y_test[:, i] == 1 # select all rows whose true label is i (boolean)
        if np.sum(rows_i) > 0:
            mean_prob_i = np.mean(y_pred_proba[rows_i, i]) # y_pred_proba[rows_i, assumed_next_state]: only calculate rows of True
        else:
            mean_prob_i = np.nan
        # print(f"Probability of truly predicting class {i} is {mean_prob_i}")
        mean_prob_class.append(mean_prob_i)

    mean_prob = np.nanmean(mean_prob_class)
    return mean_prob

# Evaluation by log probability with base `e`
# 1. sample-based: -ln() on every sample and then average
# Range: (0, ln(num_classes))
def entropy_sample(y_pred_proba, y_test):
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
# 2. class-based: -ln() on every class and then average
# similar to PTP calculation
def entropy_class(y_pred_proba, y_test):
    mean_entropy_class = []
    for i in range(y_pred_proba.shape[1]):
        rows_i = y_test[:, i] == 1
        if np.sum(rows_i) > 0:
            entropy_i = np.mean(-np.log(y_pred_proba[rows_i, i]))
        else:
            entropy_i = np.nan
        print(f"Entropy for class {i}: {entropy_i}")
        mean_entropy_class.append(entropy_i)
    mean_entropy = np.nanmean(mean_entropy_class)

    # Save class entropy result
    row = {f'class_{i}': entropy for i, entropy in enumerate(mean_entropy_class)}
    row['mean_entropy'] = mean_entropy

    df = pd.DataFrame([row])
    filename='entropy_class_log.csv'
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    
    return mean_entropy

# Evaluation by brier score
def brier(y_pred_proba, y_test):
    score_matrix = (y_pred_proba - y_test)**2
    brier_score_states = np.mean(score_matrix, axis = 0)
    # for assumed_next_state, score in enumerate(brier_score_states):
    #     print(f"Brier score for state {assumed_next_state} is {score}")
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
    # for i, score in enumerate(brier_score_states):
    #     print(f"Brier score for state {i} is {score}")
    brier_score = np.sum(brier_score_states)

    return brier_score