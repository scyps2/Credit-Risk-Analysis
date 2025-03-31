## MLP Report
The following is examinations on data of 2020Q1.  
Here state(class) 3 means delinquency state of higher than 2 months.  
### MLP with only `Previous Loan Delinquency Status`
#### Results on train set:  
**average probability = 0.49361862426499764**  
Probability of truly predicting class 0 is 0.9934451091435801  
Probability of truly predicting class 1 is 0.08125582909927707  
Probability of truly predicting class 2 is 0.16037910580976664  
Probability of truly predicting class 3 is 0.7393944530073668  
**brier score =  0.014891547048121827**  
Brier score for state 0 is 0.006567268370069974  
Brier score for state 1 is 0.005746053930023123  
Brier score for state 2 is 0.0014493765330497329  
Brier score for state 3 is 0.0011288482149789958  

#### Results on test set:  
**average probability = 0.4955193236275815**  
Probability of truly predicting class 0 is 0.9930507611570091  
Probability of truly predicting class 1 is 0.10158275795180167  
Probability of truly predicting class 2 is 0.15705111785060197  
Probability of truly predicting class 3 is 0.7303926575509132  
**brier score =  0.018360628106507296**  
Brier score for state 0 is 0.007768639160916899  
Brier score for state 1 is 0.006996330386601547  
Brier score for state 2 is 0.001963766526432156  
Brier score for state 3 is 0.001631892032556692  

### MLP with also `Credit Score`
#### Results on train set:  
**average probability = 0.49897476170817634**  
Probability of truly predicting class 0 is 0.9933947804218469  
Probability of truly predicting class 1 is 0.08205197913003032  
Probability of truly predicting class 2 is 0.18534577788141554  
Probability of truly predicting class 3 is 0.7351065093994124  
**brier score =  0.0149677445408871**  
Brier score for state 0 is 0.0065842036416194445  
Brier score for state 1 is 0.00574243031317572  
Brier score for state 2 is 0.0014435011078176383  
Brier score for state 3 is 0.0011976094782742966  

#### Results on test set:  
**average probability = 0.506295248261863**  
Probability of truly predicting class 0 is 0.9931707566305218  
Probability of truly predicting class 1 is 0.10946207335356292  
Probability of truly predicting class 2 is 0.19512208871662678  
Probability of truly predicting class 3 is 0.7274260743467407  
**brier score =  0.018458786555378436**  
Brier score for state 0 is 0.0077803780164585066  
Brier score for state 1 is 0.006929036239252714  
Brier score for state 2 is 0.001959156322055135  
Brier score for state 3 is 0.0017902159776120778  

### MLP with other five main features
```python
["Credit Score", "Original UPB", "Original Debt-to-Income (DTI) Ratio", "Original Loan-to-Value (LTV)", "Original Interest Rate"]
```
#### Results on train set:  
**average probability = 0.5009808806292595**  
Probability of truly predicting class 0 is 0.9934632066872101  
Probability of truly predicting class 1 is 0.07094718378219947  
Probability of truly predicting class 2 is 0.16993923861741422  
Probability of truly predicting class 3 is 0.7695738934302141  
**brier score =  0.014856850050616174**  
Brier score for state 0 is 0.006507416417497639  
Brier score for state 1 is 0.005742269986937353  
Brier score for state 2 is 0.001480872574023168  
Brier score for state 3 is 0.0011262910721580141  

#### Results on test set:  
**average probability = 0.49966579458923066**  
Probability of truly predicting class 0 is 0.9932778488551468  
Probability of truly predicting class 1 is 0.08476200630691624  
Probability of truly predicting class 2 is 0.17056453461949794  
Probability of truly predicting class 3 is 0.7500587885753616  
**brier score =  0.01873666606986128**  
Brier score for state 0 is 0.007594427121554932  
Brier score for state 1 is 0.007087360103429906  
Brier score for state 2 is 0.0019945901881885065  
Brier score for state 3 is 0.002060288656687935   

#### _Authored by Peini SHE on Mar.31, 2025_