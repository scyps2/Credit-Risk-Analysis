## MLP Report（Week 6）
The following is examinations on data of 2020Q1.  
Here state(class) 7 means delinquency state of higher than 6 months.  

This week we weighted brier score to punish predictions farther from true labels more. The algorithm is as below:  
```python
num_classes = y_pred_proba.shape[1]

# decode one hot
true_labels = np.argmax(y_test, axis=1)

for i, true_label in enumerate(true_labels):
    # calculate weight list
    distances = np.abs(np.arange(num_classes) - true_label)
    weights = (distances + 1) ** distance_power # default to be linear as 1
    weighted_score = weights * score_matrix[i] / np.sum(weights)
    weighted_scores.append(weighted_score)
```

### MLP with only 'Previous Deliquency Status'

**brier score =  0.021210369771324393**  
Brier score for state 0 is 0.008971940392930545  
Brier score for state 1 is 0.007354259839912155  
Brier score for state 2 is 0.002124460454478155  
Brier score for state 3 is 0.0009008098940391777  
Brier score for state 4 is 0.0005279990404296844  
Brier score for state 5 is 0.0003197047826396724  
Brier score for state 6 is 0.00029700065555930834  
Brier score for state 7 is 0.0007141947113356965  

**adjusted brier score =  0.0011948258750159376**  
Brier score for state 0 is 0.0005523822581443974  
Brier score for state 1 is 0.00026458999331092177  
Brier score for state 2 is 0.00010125147345563589  
Brier score for state 3 is 5.799362730942753e-05  
Brier score for state 4 is 3.940098770195379e-05  
Brier score for state 5 is 3.1322389776507724e-05  
Brier score for state 6 is 2.4678898897719473e-05  
Brier score for state 7 is 0.00012320624641937403  

### MLP with also `Credit Score`

**brier score =  0.02115816506891893**  
Brier score for state 0 is 0.00896251921771285  
Brier score for state 1 is 0.007324184955299592  
Brier score for state 2 is 0.002119494691205232  
Brier score for state 3 is 0.0008923010351237158  
Brier score for state 4 is 0.0005377736024357184  
Brier score for state 5 is 0.00032175626383579086  
Brier score for state 6 is 0.0002893907592798636  
Brier score for state 7 is 0.0007107445440261659  

**adjusted brier score =  0.001201311530513283**  
Brier score for state 0 is 0.0005534384834319791  
Brier score for state 1 is 0.00026247393983932485  
Brier score for state 2 is 0.00010105887430931483  
Brier score for state 3 is 5.80646612412423e-05  
Brier score for state 4 is 4.04118232794849e-05  
Brier score for state 5 is 3.215531829923827e-05  
Brier score for state 6 is 2.8674038591171322e-05  
Brier score for state 7 is 0.00012503439152152754  

#### After relabelling Credit Score

**brier score =  0.021207271891365342**  
Brier score for state 0 is 0.00898266041276095  
Brier score for state 1 is 0.007347500511825636  
Brier score for state 2 is 0.0021272615686340752  
Brier score for state 3 is 0.000893883816809593  
Brier score for state 4 is 0.0005316786331618528  
Brier score for state 5 is 0.0003200049037122438  
Brier score for state 6 is 0.0002928417604225396  
Brier score for state 7 is 0.000711440284038449  

**adjusted brier score =  0.0011909752425610478**  
Brier score for state 0 is 0.0005417449651493335  
Brier score for state 1 is 0.0002655153627284599  
Brier score for state 2 is 0.00010315654807473777  
Brier score for state 3 is 5.9580597524475654e-05  
Brier score for state 4 is 3.882149270404521e-05  
Brier score for state 5 is 3.0012328339543674e-05  
Brier score for state 6 is 2.7968134353940117e-05  
Brier score for state 7 is 0.00012417581368651188  

### MLP with other five main features
```python
["Credit Score", "Original UPB", "Original Debt-to-Income (DTI) Ratio", "Original Loan-to-Value (LTV)", "Original Interest Rate"]
```
**brier score =  0.021230246543668608**  
Brier score for state 0 is 0.008946590504328767  
Brier score for state 1 is 0.007368248910457557  
Brier score for state 2 is 0.0020834933577912004  
Brier score for state 3 is 0.0008926066098393566  
Brier score for state 4 is 0.0005461769749695848  
Brier score for state 5 is 0.00034997860223982114  
Brier score for state 6 is 0.0003135047161063727  
Brier score for state 7 is 0.0007296468679359476  

**adjusted brier score =  0.0011986162729286703**  
Brier score for state 0 is 0.0005427488996622281  
Brier score for state 1 is 0.00026603537149157337  
Brier score for state 2 is 9.914652850393491e-05  
Brier score for state 3 is 5.873477294851869e-05  
Brier score for state 4 is 4.072988133167518e-05  
Brier score for state 5 is 3.317875846868616e-05  
Brier score for state 6 is 3.017761789689515e-05  
Brier score for state 7 is 0.0001278644426251587  
