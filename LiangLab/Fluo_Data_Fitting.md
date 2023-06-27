# Fluo Data Fitting

[TOC]

Different regression methods are tried to fit the fluorescence data -- the relationship between **the green fluorescence data at the dendrite synapses** and **the red fluorescence data at the cell body**.

## Data Reading, Manipulation, and Preprocessing

After reading and preprocessing the data (.mat files), the data from all cells is stored in a population dictionary. The dictionary's keys correspond to the cell names, and the values contain the data for each respective cell.

The values in the population dictionary are cell dictionaries with keys being `'axons'`, `'green_dFFMeanValues'`,and `'red_dFFMeanValues'`:

- The value of `'axons'`, e.g., `cell_data_dict['CL090_230515']['axons']` is a 1 dimensional numpy array, of which the length is the number of groups and the elements are 1 dimensional numpy arrays consisting of components belonging to the group.
- The value of `'green_dFFMeanValues'` is a 2 dimensional 3 by 49 numpy array (each cell has 3 rounds, and each round has 8 directions \* 2 time frequencies \* 3 space frequencies = 48 settings plus a extra period so in total there are 49 columns), of which the elements are still 2 dimensional numpy arrays with size being 10 by N (N is the number of components).
- The value of `'red_dFFMeanValues'` is similarly a 2 dimensional 3 by 49 numpy array, of which the elements are still 2 dimensional numpy arrays with size being 10 by 1 (only recording the data at the soma).

Note: 

- Four cells: `'CL090_230515'`, `'CL090_230518'`, `'CL083_230413'`, `'CL075_230303'`.
- `'red_dFFMeanValues'` and `'green_dFFMeanValues'` have 49 columns,where the last column should be excluded. They are supposed to have 3 rows (3 rounds), but `'CL090_230518'` only has 2 rows.
- In `'CL083_230413'`, elements in `'red_dFFMeanValues'` have 2 columns (10 × 2， should be 10 × 1), so `'CL083_230413'` is not used.

**Sample**: `'CL090_230515'`.

**Data shape**:

|                    | Train data                | Test data              |
| ------------------ | ------------------------- | ---------------------- |
| Predictors (green) | x_train shape: (1368, 25) | x_test shape: (72, 25) |
| Targets (red)      | y_train shape: (1368,)    | y_test shape: (72,)    |

## Linear Regression

### Ordinary Linear Regression

Ordinary least squares Linear Regression.

Linear Regression fits a linear model with coefficients to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

**Fitted parameters**:

Fitted Coefficients: [ 8.28301566e-03  7.56874223e-03  2.17645374e-04 -2.06269935e-03
  3.53903502e-03  5.16754189e-03  1.24691680e-03  2.32002780e-02
  2.40241470e-03 -6.16858257e-03 -1.20710729e-03  2.78029746e-02
  9.57603350e-03 -4.35972960e-03  3.04637298e-03  6.77257749e-03
  8.29508492e-04  3.04802829e-02 -6.67335217e-04  9.61650591e-03
 -2.47112388e-02 -1.14696165e-03  8.29246943e-02 -1.13145597e-02
 -3.19379843e-01]

Fitted Intercept: -0.003966473464240797

Some coefficients are negative. **Is it reasonable???** Explanation: higher concentration of glutamate can mean even higher positively correlated concentration of GABA; two pathways, one is pre-cell to current synapsis, the other is pre-cell to an inhibitory cell then to the current synapsis (disinhibition).

**Results**:

|                         | Train data            | Test data            |
| ----------------------- | --------------------- | -------------------- |
| Mean squared error      | 0.0061596970599993445 | 0.007927335708322457 |
| Correlation coefficient | 0.6571303587882936    | 0.6140976886111208   |

![Comparison of Sorted Predictions and Sorted Ground Truth (Linear Regression, Train Set)](pics\Comparison of Sorted Predictions and Sorted Ground Truth (Ordinary Linear Regression, Train Set).png)

![Comparison of Sorted Predictions and Sorted Ground Truth (Linear Regression, Test Set)](pics\Comparison of Sorted Predictions and Sorted Ground Truth (Ordinary Linear Regression, Test Set).png)

### Ridge Linear Regression

Linear least squares with l2 regularization.

Minimizes the objective function:

$$
||y - Xw||^2_2 + \alpha  ||w||^2_2
$$

This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm.

**Fitted parameters**:

Fitted Coefficients: [ 0.00849453  0.00740115  0.00016157 -0.0020944   0.0034469   0.00444959  0.00196809  0.0228857   0.0024516  -0.00571464 -0.00091334  0.0277737  0.00929144 -0.00475298  0.00350069  0.00617136 -0.00020173  0.02975458 -0.00172438  0.01062452 -0.02926797 -0.00329156  0.01071774 -0.01549336 -0.06836985] 

Fitted Intercept: -0.003992042396819981

**Results**:

|                         | Train data          | Test data            |
| ----------------------- | ------------------- | -------------------- |
| Mean squared error      | 0.00617267529320373 | 0.008058648873285792 |
| Correlation coefficient | 0.6562194390940245  | 0.6047541953182045   |

![Comparison of Sorted Predictions and Sorted Ground Truth (Linear Regression, Train Set)](pics\Comparison of Sorted Predictions and Sorted Ground Truth (Ridge Linear Regression, Train Set).png)

![Comparison of Sorted Predictions and Sorted Ground Truth (Linear Regression, Train Set)](pics\Comparison of Sorted Predictions and Sorted Ground Truth (Ridge Linear Regression, Test Set).png)

### Elasticnet Linear Regression

Linear regression with combined L1 and L2 priors as regularizer.

Minimizes the objective function:

$$
1 / (2 * n_{samples}) * ||y - Xw||^2_2
+ \alpha * l1_{ratio} * ||w||_1
+ 0.5 * \alpha * (1 - l1_{ratio}) * ||w||^2_2
$$

If controlling the L1 and L2 penalty separately, that this is equivalent to:

$$
a * ||w||_1 + 0.5 * b * ||w||_2^2
$$

where: $\alpha = a + b$ and $l1_{ratio} = a / (a + b)$.

Let $a=0.004$ and $b=0$, non-negative coefficients are achieved. **BUT it cannot be positive if it is negative without penalty.**

**Fitted parameters**:

Fitted Coefficients: [ 0.00951724  0.0068485   0.         -0.          0.00285676  0.  0.          0.01545406  0.00635406 -0.          0.          0.02601547  0.00529342 -0.          0.          0.00326147  0.          0.  0.          0.         -0.          0.          0.         -0. -0.        ] 

Fitted Intercept: -0.004587622786060119

**Results**:

|                         | Train data           | Test data            |
| ----------------------- | -------------------- | -------------------- |
| Mean squared error      | 0.006385274236585692 | 0.008246618904776503 |
| Correlation coefficient | 0.6428087118878939   | 0.593481482809575    |

![Comparison of Sorted Predictions and Sorted Ground Truth (Linear Regression, Train Set)](pics\Comparison of Sorted Predictions and Sorted Ground Truth (Elasticnet Linear Regression, Train Set).png)

![Comparison of Sorted Predictions and Sorted Ground Truth (Linear Regression, Train Set)](pics\Comparison of Sorted Predictions and Sorted Ground Truth (Elasticnet Linear Regression, Test Set).png)
