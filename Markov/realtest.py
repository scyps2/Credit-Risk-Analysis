import  numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv('simCRdata.csv')
df = df.sort_values(by = ['cust','t'])
print(df.head())

df['y_next'] = df.groupby('cust')['y'].shift(-1)
df['y_prev'] = df.groupby('cust')['y'].shift(1)
df_clean = df.dropna(subset = ['y_next','y_prev'])

contingency_table = pd.crosstab(index = [df_clean['y_prev'],df_clean['y']], columns = df_clean['y_next'])
transition_matrix = pd.crosstab(df_clean['y'], df_clean['y_next'], normalize = 'index')

print(contingency_table)
print(transition_matrix)

# Perform chi-square test
chi2, p, dof, ex = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")

# Interpret the result
alpha = 0.05
if p > alpha:
    print("First-order assumption holds (fail to reject H0)")
else:
    print("First-order assumption does not hold (reject H0)")
    
# 展示转移矩阵（表示从一种状态转移到另一种状态的概率）
print("Transition matrix (Markov model):")
print(transition_matrix)

# 示例：查看状态 1（逾期 1 个月）的转移概率
print("\ntransition probability from state 1 is-:")
print(transition_matrix.loc[1])
'''----------------------------- find the transition matrix'''
'--------find the steady-state distribution--------'
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

# Find the eigenvector corresponding to eigenvalue 1
steady_state = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten().real

# Normalize the steady-state vector to sum to 1
steady_state /= steady_state.sum()

print("Steady-State Distribution:")
print(steady_state)


