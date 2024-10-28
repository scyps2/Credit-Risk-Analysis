import  numpy as np
import pandas as pd
df_train=pd.read_csv('simCRdata_train2.csv')
df_train=df_train.sort_values(by=['cust','t'])
df_test=pd.read_csv('simCRdata_test2.csv')
df_test=df_test.sort_values(by=['cust','t'])

grouped = df_train.groupby('grade')
grade_0_data = grouped.get_group(0)
grade_1_data = grouped.get_group(1)

# grade_0_data=df_train[df_train['grade'] == 0].copy()
# grade_1_data=df_train[df_train['grade'] == 1].copy()
# grade_0_data.loc[:,'y_next']=grade_0_data.groupby('cust')['y'].shift(-1)
# grade_1_data.loc[:,'y_next']=grade_1_data.groupby('cust')['y'].shift(-1)

grade_0_data['y_next']=grade_0_data.groupby('cust')['y'].shift(-1)
grade_1_data['y_next']=grade_1_data.groupby('cust')['y'].shift(-1)

df_0_clean= grade_0_data.dropna(subset=['y_next'])
df_1_clean= grade_1_data.dropna(subset=['y_next'])
######generate 2 transition matrix 
transition_matrix_grade_0 = pd.crosstab(df_0_clean['y'], df_0_clean['y_next'], normalize='index')
transition_matrix_grade_1 = pd.crosstab(df_1_clean['y'], df_1_clean['y_next'], normalize='index')
print('the transition matrix of grade 0:')
print(transition_matrix_grade_0)
print('the transition matrix of grade 1:')
print(transition_matrix_grade_1)
####### calculate the brier score 
grade_0_test=df_test[df_test['grade']==0].copy()
grade_1_test=df_test[df_test['grade']==1].copy()
grade_0_test.loc[:,'y_next']=grade_0_test.groupby('cust')['y'].shift(-1)
grade_1_test.loc[:,'y_next']=grade_1_test.groupby('cust')['y'].shift(-1)
df_0_clean_test= grade_0_test.dropna(subset=['y_next']).copy()
df_1_clean_test= grade_1_test.dropna(subset=['y_next']).copy()
def brier_score_multiclass(predicted_probs,actual):
    actual_one_hot=np.zeros_like(predicted_probs)
    actual_one_hot[actual]=1
    return np.sum((predicted_probs-actual_one_hot)**2)
brier_scores_grade0=[]
brier_scores_grade1=[]
for index,row in df_0_clean_test.iterrows():
    current_state=int(row['y'])
    next_state=int(row['y_next'])
    predicted_probs=transition_matrix_grade_0.loc[current_state]
    score=brier_score_multiclass(predicted_probs,next_state)
    brier_scores_grade0.append(score)
df_0_clean_test.loc[:,'brier score']=brier_scores_grade0
for index,row in df_1_clean_test.iterrows():
    current_state=int(row['y'])
    next_state=int(row['y_next'])
    predicted_probs=transition_matrix_grade_1.loc[current_state]
    score=brier_score_multiclass(predicted_probs,next_state)
    brier_scores_grade1.append(score)
df_1_clean_test.loc[:,'brier score']=brier_scores_grade1

df_0_clean_test.to_csv('grade0_test2_brier_score.csv',index=False)
df_1_clean_test.to_csv('grade1_test2_brier score.csv',index=False)
average_brier_score_grade0=np.mean(brier_scores_grade0)
average_brier_score_grade1=np.mean(brier_scores_grade1)
print('brier score of grade 0 = ')
print(average_brier_score_grade0)
print('brier score of grade 1 = ')
print(average_brier_score_grade1)
    


