# 2024 Autumn Project on Credit Risk
This is a project aiming at analyzing overdue possibilities of customer credit, using Markov Chain and Neuro Network. Starting from 2024 autumn.

## Works before joining in
Markov Chain model, see first commit of Markov Folder.  
**brier score = 0.12729008977196365**  
brier score when transition matrix is Identity matrix = 0.23454545454545456  
_** Identity matrix: state transition doesn't happen. The next state copies current state for 100% probability._  
 
## Week 1 (Oct. 14)
### Tasks
- [x] Build multi-class MLPClassifier with scikit-learn, predict on 'y_next'
- [x] Evaluate results on brier score
- [x] Draw ROC curve
### Outcomes
Basic ANN, with current month overdue state 'y' as input(one-hot encoded). Two hidden layers, each with 10 neurons. Output a probability matrix for all states of next month.  
 
Brier score for state 0 is 0.026767212942569212  
Brier score for state 1 is 0.03999608844442646  
Brier score for state 2 is 0.026211268859962226  
Brier score for state 3 is 0.03474007548554805  
**brier score = 0.12771464573250563**  
 
![week 1 ROC](MLP/figs/basic.png)  
Best threshold for state 0 is  0.01  
Best threshold for state 1 is  0.04  
Best threshold for state 2 is  0.06  
Best threshold for state 3 is  0.29  
_** Threshold: If probability for state i > threshold, this is considered as 'positive'. Otherwise 'negative'. Best means we achieve the highest TRF._  

## Week 2 (Oct. 21)
### Tasks
- [x] Include 'grade' to NN
- [x] Wait for larger dataset and test again
- [x] Test on training dataset
### Outcomes
Result gets worse after including 'grade'.  
 
Brier score for state 0 is 0.026593834862040974  
Brier score for state 1 is 0.03991571728797261  
Brier score for state 2 is 0.02782713063525248  
Brier score for state 3 is 0.03674043336080793  
brier score = 0.13107711614607478  
 
![week 2 ROC](MLP/figs/grade.png)  
Best threshold for state 0 is  0.01  
Best threshold for state 1 is  0.04  
Best threshold for state 2 is  0.06  
Best threshold for state 3 is  0.20  
#
Results improve with larger trainging dataset, especially those with overdue records.  
#### Markov 
brier score of grade 0 = 0.13982951711633454    
brier score of grade 1 = 0.0867008525177791  
**average =  0.11326518481705683**

#### MLP 
Brier score for state 0 is 0.038412409361005886  
Brier score for state 1 is 0.038244722036273104  
Brier score for state 2 is 0.013505776922025611  
Brier score for state 3 is 0.009791743234580331  
**brier score = 0.09995465155388493 (with grade)**  
brier score = 0.10098478835919814 (without grade)  

![week 2 ROC](MLP/figs/larger_dataset.png)  
Best threshold for state 0 is  0.01  
Best threshold for state 1 is  0.01  
Best threshold for state 2 is  0.02  
Best threshold for state 3 is  0.01  
# 
Test on training dataset, slightly better result.  
brier score = 0.09927848251748608  

## Week 3 (Oct. 28)
### Tasks
- [x] Get auc value of ROC
- [ ] Train with smaller dataset spliting from data2
- [ ] Calculate average on several random states
- [ ] Try to adjust NN parameters / apply advanced spliting methods, expected to reach Markov average

### Outcomes
Original results with dataset3:
#### Markov
brier score of grade 0 = 0.14363123495884136      
brier score of grade 1 = 0.5709526434983646    
**average = 0.357291939228603**

#### MLP
Brier score for state 0 is 0.17368632949775978  
Brier score for state 1 is 0.17616818985984314  
Brier score for state 2 is 0.08629849458531651  
Brier score for state 3 is 0.02457887700079036  
**brier score = 0.4607318909437098**
![week 3 ROC](MLP/figs/dataset3.png) 
