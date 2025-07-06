## MLP Report（Week 8）
Here we built **separate MLP models for previous deliquency state is 0 and not 0**.
Since Iteration method 1 (remain probability form) in Report 7 performs the best, we'll continue on it in this report. The results shows the performance on predicting 6 month ahead.

#### PTP
average probability = 0.12500313946055447  
average probability for previous state 0 = 0.12502899907003462  
average probability for previous state 1 = 0.1253188271357427  

#### BS
brier score = 0.036350595007788845  
brier score for previous state 0 = 0.018900039750736147  
brier score for previous state 1 = 0.6356853340477031  

adjusted brier score =  0.0033003094665204243  
adjusted brier score for previous state 0 =  0.0014556793801633343  
adjusted brier score for previous state 1 =  0.05605353159664106  

#### Entropy
entropy_sample = 0.1403001413126807  
entropy_sample for previous state 0 = 0.12502899907003462  
entropy_sample for previous state 1 = 1.2824931267568203  

entropy_class = 6.1744574477470735  
```python
Entropy for class 0: 0.011114068045935045  
Entropy for class 1: 5.125232762787922  
Entropy for class 2: 6.573120663375041  
Entropy for class 3: 7.0099463438559155  
Entropy for class 4: 6.956289795701771  
Entropy for class 5: 6.901540445207243  
Entropy for class 6: 7.115297263671631  
Entropy for class 7: 9.703118239331125  
```

entropy_class for previous state 0 = 6.330252686296429  
```python
Entropy for class 0: 0.009684474645028697  
Entropy for class 1: 5.281069348567585  
Entropy for class 2: 6.700859496910161  
Entropy for class 3: 7.048222820667606  
Entropy for class 4: 7.011617237082056  
Entropy for class 5: 6.910382331532179  
Entropy for class 6: 7.185459076359963  
Entropy for class 7: 10.494726704606851  
```

entropy_class for previous state 1 = 3.134117463734842  
```python
Entropy for class 0: 0.77911213074927  
Entropy for class 1: 3.094794910568891  
Entropy for class 2: 3.3616536761896203  
Entropy for class 3: 3.722517670492367  
Entropy for class 4: 4.143062615820376  
Entropy for class 5: 4.60639517766667  
Entropy for class 6: 4.454380139948421  
Entropy for class 7: 0.9110233884431179  
```

### With `Credit Score` 
#### PTP
average probability = 0.1260424058547386  
average probability for previous state 0 = 0.12604004519510004  
average probability for previous state 1 = 0.13094588845031802  

#### BS
brier score = 0.03612290225110014  
brier score for previous state 0 = 0.018822760236138814  
brier score for previous state 1 = 0.6093161340585304  

adjusted brier score = 0.003264141786604554  
adjusted brier score for previous state 0 = 0.0014466343746454987  
adjusted brier score for previous state 1 = 0.050159654852341404  

#### Entropy
entropy_sample = 0.13893579041996515  
entropy_sample for previous state 0 = 0.12604004519510004  
entropy_sample for previous state 1 = 1.213914305140022  

entropy_class = 6.006410253040569  
```python
Entropy for class 0: 0.011007302882375829
Entropy for class 1: 4.8854252830174065
Entropy for class 2: 6.09151712530423
Entropy for class 3: 6.672550690730655
Entropy for class 4: 6.746099006838859
Entropy for class 5: 6.636724084199539
Entropy for class 6: 6.857095597443114
Entropy for class 7: 10.150862933908375
```
entropy_class for previous state 0 = 5.903951938171852
```python
Entropy for class 0: 0.009855370000773142
Entropy for class 1: 4.974158262498801
Entropy for class 2: 6.334141331547897
Entropy for class 3: 6.7913535605224045
Entropy for class 4: 6.893299074670984
Entropy for class 5: 6.757158293211548
Entropy for class 6: 6.961890798317221
Entropy for class 7: 8.509758814605195
```
entropy_class for previous state 1 = 2.850907243017002
```python
Entropy for class 0: 0.6816301744952827
Entropy for class 1: 2.5740020898541056
Entropy for class 2: 3.045744146732659
Entropy for class 3: 3.3467588144952654
Entropy for class 4: 3.931577838290291
Entropy for class 5: 3.9944112344693496
Entropy for class 6: 4.0812252131374045
Entropy for class 7: 1.1519084326616584
```

### With 5 main features
#### PTP
average probability = 0.12636797619385623  
average probability for previous state 0 = 0.12621295097956953  
average probability for previous state 1 = 0.1341381866873806  

#### BS
brier score = 0.0361114863160904  
brier score for previous state 0 = 0.01880682230651717  
brier score for previous state 1 = 0.627885492724681  

adjusted brier score = 0.0032507267572834645  
adjusted brier score for previous state 0 = 0.0014400320664585188  
adjusted brier score for previous state 1 = 0.05761209309827995  

#### Entropy
entropy_sample = 0.13737027930305218  
entropy_sample for previous state 0 = 0.12621295097956953  
entropy_sample for previous state 1 = 1.2275145153944702  

entropy_class = 5.875203359597103  
```python
Entropy for class 0: 0.011707347337098072
Entropy for class 1: 4.791874453510502
Entropy for class 2: 6.0115629004396895
Entropy for class 3: 6.5531863987565755
Entropy for class 4: 6.566625715678599
Entropy for class 5: 6.4439750611268325
Entropy for class 6: 6.600455014045901
Entropy for class 7: 10.022239985881622
```
entropy_class for previous state 0 = 5.848519403013594
```python
Entropy for class 0: 0.009912620052847754
Entropy for class 1: 4.914108899316138
Entropy for class 2: 6.274279990480851
Entropy for class 3: 6.796749467833685
Entropy for class 4: 6.698908125464824
Entropy for class 5: 6.481659325000567
Entropy for class 6: 6.710905515516415
Entropy for class 7: 8.901631280443425
```
entropy_class for previous state 1 = 3.0756468783724613  
```python
Entropy for class 0: 0.7265890052887436
Entropy for class 1: 2.8850539676935303
Entropy for class 2: 3.415815418632042
Entropy for class 3: 3.8712640678560253
Entropy for class 4: 4.26081502633103
Entropy for class 5: 4.217852166190758
Entropy for class 6: 4.339587099020476
Entropy for class 7: 0.8881982759670864
```