# Project on Credit Risk

Author: Peini SHE

## Symbols

| Glossary | Meaning                                                      | Type   |
| -------- | ------------------------------------------------------------ | ------ |
| y        | overdue state of current month                               | int    |
| y_next   | overdue state of next month                                  | int    |
| grade    | customer credit rating                                       | binary |
| mev      | a measure of macroeconomic conditions at each transaction step | binary |

## Results

_** brier score in bold means the major result for a certain dataset._

### Dataset 1

| MLP inputs       | MLP parameters                                               | Brier score             | Explanation                                |
| ---------------- | ------------------------------------------------------------ | ----------------------- | ------------------------------------------ |
| y, y_next        | `hidden_layer_sizes = (10, 10), activation = 'relu', max_iter = 500, random_state = 1, learning_rate_init = 0.01, learning_rate = 'adaptive'` | **0.12753418871545052** | /                                          |
| y, y_next, grade | same as previous                                             | 0.13148802373039684     | Result gets worse after including `grade`. |

### Dataset 2

Improvement: Larger, especially more samples for `y`(`y_next`) > 1

| MLP inputs       | MLP parameters   | Brier score             | Explanation                                                  |
| ---------------- | ---------------- | ----------------------- | ------------------------------------------------------------ |
| y, y_next, grade | same as previous | **0.09996880733177178** | Results improve with larger training dataset, especially those with overdue records. |
| y, y_next        | same as previous | 0.10173659080949476     | For larger dataset, including `grade` has positive performance. |

### Dataset 3

Improvement: more distinct transition matrix to generate dataset

| MLP inputs       | MLP parameters   | Brier score         | Explanation |
| ---------------- | ---------------- | ------------------- | ----------- |
| y, y_next, grade | same as previous | 0.46093295051842986 | /           |

### Dataset 4

Improvement: extra variables: `mev`, `start_date`, `var`

| MLP inputs                 | MLP parameters                                               | Brier score            | Explanation                                                  |
| -------------------------- | ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ |
| y, y_next, grade           | `hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 500, random_state = 1, learning_rate_init = 0.0001, learning_rate = 'adaptive'` | 0.48201010929602456    | /                                                            |
| y, y_next, grade, mev      | same as previous                                             | **0.4821929395302732** | After including `mev`, result slightly gets worse.           |
| y, y_next, grade, mev      | same as previous                                             | 0.48273539124954307    | After standarizing `mev`, result gets worse.                 |
| y, y_next, grade, var      | same as previous                                             | 0.48334336283187024    | Dummy variable successfully trick the MLP.                   |
| y, y_next, grade, mev, var | same as previous                                             | 0.48249568347898775    | `mev` has more positive influence than `var`.                |
| y, y_next, grade, mev      | `'hidden_layer_sizes': (20, 20, 20), 'activation': 'relu', 'max_iter': 1000,  'learning_rate_init': 0.001, 'learning_rate': 'adaptive', ` | 0.48249191051984697    | Optimize parameters using GridSearchCV, the best result is slightly worse than non-optimization. |

### Dataset 5

Improvement: `mev` set to binary (-1 or +1)

| MLP inputs            | MLP parameters                                               | Brier score                                                  | Explanation                                            |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| y, y_next, grade      | `hidden_layer_sizes = (8, ), activation = 'relu', max_iter = 5000, random_state = 1, learning_rate_init = 0.0001, learning_rate = 'adaptive'` | 0.5280776407016857 (overall) 0.28869645437008956 (grade 0) 0.5885267523529792 (grade 1) | /                                                      |
| y, y_next, grade, mev | same as previous                                             | 0.5134315398585529 (overall) 0.2783728062786166 (grade 0) 0.5813185724896273 (grade 1) | In this dataset, mev starts to contribute.             |
| y, y_next, grade, mev | `hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 2000, random_state = 1, learning_rate_init = 0.0001, learning_rate = 'adaptive'` | 0.5094424710423735 (overall) 0.2774359851835344 (grade 0) 0.5797891729547955 (grade 1) | Adjust hidden layer size, result slightly gets better. |

## Summary

Overall, variables `grade`, `mev` has positive influence on the MLP, and the brier score doesn't relate much with MLP parameters. However, `hidden_layer_sizes` and `learning_rate_init` can slightly impact on the result.  

For the last step in dataset 5, we have weighted average for grade 0 and grade 1 = 0.4869375. This is better than not separating by grade to do the classification.