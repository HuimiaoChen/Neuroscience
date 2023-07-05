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

The cell used in this report is **Sample**: `'CL090_230515'`.

**Data shape** of **Sample**: `'CL090_230515'`:

|                    | Train data                | Test data              |
| ------------------ | ------------------------- | ---------------------- |
| Predictors (green) | x_train shape: (1368, 25) | x_test shape: (72, 25) |
| Targets (red)      | y_train shape: (1368,)    | y_test shape: (72,)    |

## Linear Regression

### Introduction

Three linear regression methods are used here. As the penalty increases, the performance slightly drops but the weights become small and sparse.

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
| R2 score                | 0.43182030844123154   | 0.3610848833977214   |
| Mean squared error      | 0.0061596970599993445 | 0.007927335708322457 |
| Correlation coefficient | 0.6571303587882936    | 0.6140976886111208   |

![Comparison (Linear Regression, Train Set)](pics_CL090_230515\Comparison (Ordinary Linear Regression, Train Set).png)

![Comparison (Linear Regression, Test Set)](pics_CL090_230515\Comparison (Ordinary Linear Regression, Test Set).png)

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
| R2 score                | 0.43062317675322503 | 0.35050150845424777  |
| Mean squared error      | 0.00617267529320373 | 0.008058648873285792 |
| Correlation coefficient | 0.6562194390940245  | 0.6047541953182045   |

![Comparison (Linear Regression, Train Set)](pics_CL090_230515\Comparison (Ridge Linear Regression, Train Set).png)

![Comparison (Linear Regression, Train Set)](pics_CL090_230515\Comparison (Ridge Linear Regression, Test Set).png)

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
| R2 score                | 0.4110127314829676   | 0.33535179119658987  |
| Mean squared error      | 0.006385274236585692 | 0.008246618904776503 |
| Correlation coefficient | 0.6428087118878939   | 0.593481482809575    |

![Comparison (Linear Regression, Train Set)](pics_CL090_230515\Comparison (Elasticnet Linear Regression, Train Set).png)

![Comparison (Linear Regression, Train Set)](pics_CL090_230515\Comparison (Elasticnet Linear Regression, Test Set).png)

## Power-Law Regression

### Introduction

Mathematically, a power-law relationship can be expressed as:

$$
y = A  X^C
$$

Here, I modify it, shown as:

$$
y = A  (X+B)^C + D
$$

where, $X = \beta_1 x_1 + \beta_2 x_2 \dots + \beta_N x_N$. $X+D$ is a linear regression part. $A$, $B$, $C$, $D$, $\beta_1$, $\beta_2$, ...,$\beta_N$ are parameters to be determined.

### Example: Exponent = 5

$C=5$.

**Fitted Parameters $A$, $B$, $D$, $\beta_1$, $\beta_2$, ...,$\beta_N$** : 

[ 3.63589934e+00  5.17800351e-01 -1.38641935e-01  4.83492282e-03  4.37660663e-03 -3.27563273e-04 -1.65395632e-05  2.76270197e-03  3.60564181e-03  1.18571822e-03  1.39874639e-02  9.74478013e-04 -4.33905140e-03  3.12368809e-04  1.63689641e-02  5.70510078e-03 -3.84107281e-03  9.86178700e-04  3.44528993e-03  1.88010948e-04  1.79227711e-02 -6.87188789e-03  6.36542080e-03 -1.91319680e-02 -5.91287026e-04  3.77753461e-02 -1.06618770e-02 -1.74782336e-01]

**Results**:

|                         | Train data           | Test data            |
| ----------------------- | -------------------- | -------------------- |
| R2 score                | 0.44977053881398854  | 0.3711914289705994   |
| Mean squared error      | 0.005965096684632098 | 0.007801938801087381 |
| Correlation coefficient | 0.6706493415945699   | 0.6219617870348478   |

![Comparison (Power-Law Regression (Exponent=5), Test Set)](pics_CL090_230515\Comparison (Power-Law Regression Exponent=5, Test Set).png)

![Comparison (Power-Law Regression (Exponent=5), Test Set)](pics_CL090_230515\Comparison (Power-Law Regression Exponent=5, Train Set).png)

### Example: Exponent = 5 (only fit A and D)

$C = 5$.

$\beta_1$, $\beta_2$, ...,$\beta_N$ given by Ordinary Linear Regression.

Fitted Coefficients: [ 8.28301566e-03  7.56874223e-03  2.17645374e-04 -2.06269935e-03  3.53903502e-03  5.16754189e-03  1.24691680e-03  2.32002780e-02  2.40241470e-03 -6.16858257e-03 -1.20710729e-03  2.78029746e-02  9.57603350e-03 -4.35972960e-03  3.04637298e-03  6.77257749e-03  8.29508492e-04  3.04802829e-02 -6.67335217e-04  9.61650591e-03 -2.47112388e-02 -1.14696165e-03  8.29246943e-02 -1.13145597e-02 -3.19379843e-01]

**Fitted Parameters $A$, $D$**: 

[3.06877195e+02 3.73110121e-02]

**Results**:

|                          | Train data          | Test data            |
| ------------------------ | ------------------- | -------------------- |
| R2 score                 | 0.16867571360029543 | 0.16149921366803333  |
| Mean squared error       | 0.00901247587500699 | 0.010403693780630378 |
| Correlation coefficient\* | 0.41070148964947206 | 0.4740522878239656   |

\* Correlation coefficient is not a good metric for nonlinear regression so that we see the number of Test data is larger than that of Train data.

![Comparison (Power-Law Regression (Exponent=5), Test Set)](pics_CL090_230515\Comparison (Power-Law Regression Exponent=5, only fit A and D, Test Set).png)

![Comparison (Power-Law Regression (Exponent=5), Test Set)](pics_CL090_230515\Comparison (Power-Law Regression Exponent=5, only fit A and D, Train Set).png)

Basically, only fitting $A$, $D$ fails to provide a good result.

### Different Exponents

#### Results

**Exponent: 1/95**

|                         | Train data          | Test data            |
| ----------------------- | ------------------- | -------------------- |
| R2 score                | 0.03578426304422133 | -0.01178350132389605 |
| Mean squared error      | 0.01045316636333401 | 0.012553698090272793 |
| Correlation coefficient | 0.4156817350556336  | 0.5492586362659945   |

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=1 over 95, Test Set).png)

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=1 over 95, Train Set).png)

**Exponent: 30/43**

|                         | Train data           | Test data            |
| ----------------------- | -------------------- | -------------------- |
| R2 score                | 0.43176358007773163  | 0.36105993515151313  |
| Mean squared error      | 0.006160312058280804 | 0.007927645253546677 |
| Correlation coefficient | 0.6570871953391387   | 0.6140764914279518   |

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=30 over 43, Test Set).png)

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=30 over 43, Train Set).png)

**Exponent: 179/65**

|                         | Train data            | Test data             |
| ----------------------- | --------------------- | --------------------- |
| R2 score                | 0.4496429961042646    | 0.3719683808539297    |
| Mean squared error      | 0.0059664793888465975 | 0.0077922987749738555 |
| Correlation coefficient | 0.6705542454661818    | 0.6224739461963064    |

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=179 over 65, Test Set).png)

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=179 over 65, Train Set).png)

**Exponent: 221/33**

|                         | Train data          | Test data            |
| ----------------------- | ------------------- | -------------------- |
| R2 score                | 0.44976921796448766 | 0.37100964253069046  |
| Mean squared error      | 0.0059651110041034  | 0.007804194315316004 |
| Correlation coefficient | 0.6706483564323005  | 0.6218374974348415   |

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=221 over 33, Test Set).png)

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=221 over 33, Train Set).png)

**Exponent: 219/23**

|                         | Train data           | Test data           |
| ----------------------- | -------------------- | ------------------- |
| R2 score                | 0.44976054121933307  | 0.37087312756007484 |
| Mean squared error      | 0.005965205069629524 | 0.00780588812404452 |
| Correlation coefficient | 0.6706418874831295   | 0.6217442394863175  |

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=219 over 23, Test Set).png)

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=219 over 23, Train Set).png)

**Exponent: 300/17**

|                         | Train data          | Test data            |
| ----------------------- | ------------------- | -------------------- |
| R2 score                | 0.449745532921901   | 0.37070570241504563  |
| Mean squared error      | 0.00596536777619391 | 0.007807965450587863 |
| Correlation coefficient | 0.6706306978729792  | 0.6216287490698504   |

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=300 over 17, Test Set).png)

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=300 over 17, Train Set).png)

**Exponent: 73/3**

|                         | Train data           | Test data            |
| ----------------------- | -------------------- | -------------------- |
| R2 score                | 0.4497396097065115   | 0.37061959954704493  |
| Mean squared error      | 0.005965431990408115 | 0.007809033771437306 |
| Correlation coefficient | 0.6706262817005308   | 0.6215582319310625   |

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=73 over 3, Test Set).png)

![Comparison (Power-Law Regression](pics_CL090_230515\Comparison (Power-Law Regression Exponent=73 over 3, Train Set).png)

#### Comparison

As the exponent increases, we can see the R2 score on longer increases more (saturation).

See the trend of R2 score below.

![Power_Law_r2_scores_plot](pics_CL090_230515\Power_Law_r2_scores_plot.png)

## Exponential Regression

### Introduction

Let $B = (b_1, b_2, \ldots , b_N)$.
$$
y = A \cdot e^{(b_1 \cdot x_1 + \ldots + b_N \cdot x_N)} + C
$$

### Results

**Results**:

|                         | Train data           | Test data            |
| ----------------------- | -------------------- | -------------------- |
| R2 score                | 0.44972294262878365  | 0.3705444716933213   |
| Mean squared error      | 0.005965612679987845 | 0.007809965919858963 |
| Correlation coefficient | 0.6706138550826338   | 0.6215183575736783   |

![Comparison (Exponential Regression)](pics_CL090_230515\Comparison (Exponential Regression, Test Set).png)

![Comparison (Exponential Regression)](pics_CL090_230515\Comparison (Exponential Regression, Train Set).png)

Exponential regression's performance is very close to that of power-law regression. Both cannot very well fit the end part, and cannot fit the beginning part.

## Logistic Regression

### Introduction

We use [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

For `sklearn.linear_model.LogisticRegression`, there is an argument called `multi-class` (see official description below), which determines the method used for multi-classification.

> multi_class{'auto', 'ovr', 'multinomial'}, default='auto'
If the option chosen is 'ovr', then a binary problem is fit for each label. For 'multinomial' the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. 'multinomial' is unavailable when solver='liblinear'. 'auto' selects 'ovr' if the data is binary, or if solver='liblinear', and otherwise selects 'multinomial'.

**OvR**

In logistic regression, the "One-vs-Rest" (OvR) approach is a common strategy **for handling multi-class classification problems**. In this approach, multiple binary logistic regression models are trained, each one considering one class as the positive class and the rest of the classes as the negative class.

After calculating the sigmoid probabilities for each class in the OvR approach, you can normalize the probabilities to ensure that their sum is equal to 1 (that's what `sklearn.linear_model.LogisticRegression` does for multi-class). Here's how you can do it:

1. Calculate the sigmoid probabilities for each class using the logistic regression model.

   The key point of logistic regression is that **we model the log of the odds ($O =\frac{p}{1-p}$) as linear**. This is called logistic regression.
   $$
   \eta=\mathrm{logit}(p)=\mathrm{log}\left(\frac{p}{1-p}\right)
   $$
   
   $$
   p= \mathrm{expit}(\eta)=\frac{e^\eta}{e^\eta+1}= \frac{1}{1+e^{-\eta}}
   $$
   
   where logit and expit are inverse functions of each other. The logit function maps probabilities (values between 0 and 1) to real numbers (values between negative infinity and positive infinity). The expit function is a sigmoid function that maps any real-valued number to the range of 0 to 1.
   
   $$
   \eta=\mathrm{logit}(p)=\mathrm{log}\left(\frac{p}{1-p}\right)
   $$
   
   $$
   p= \mathrm{expit}(\eta)=\frac{e^\eta}{e^\eta+1}= \frac{1}{1+e^{-\eta}}
   $$
   
   For class $i$ and data $\boldsymbol{X}$, $\eta$ is considered as a linear combination (if we set intercept is True, then we have $\beta_{i,0}$) as follows.

   $$
   \begin{array}{l}
   \eta_i(\boldsymbol{X})&=\mathrm{logit}\left\{P(Y=y_i\mid \boldsymbol{X})\right\}\\
   &=\mathrm{logit}\left\{P(Y=y_i\mid X_1,X_2,\dots,X_N)\right\}\\
   &=\beta_{i,0}+\beta_{i,1}X_1+\beta_{i,2}X_2+\dots+\beta_{i,N}X_N
   \end{array}
   $$
   
   All $\beta_{i,n}$ forms an $I$ (number of classes) by $N$ (input length, i.e., the number of features) matrix, resulting in a 2D matrix of probabilities for each class and each piece of input data.

2. Normalize the probabilities by dividing each probability by the sum of all probabilities (for each piece of data $\boldsymbol{X}$):

   normalized_probs = probabilities / sum(probabilities)

   This step ensures that the sum of the probabilities for all classes is equal to 1.

Note that this normalization step is necessary because the sigmoid probabilities for each class are calculated independently, treating each class as a separate binary classification problem. Normalizing the probabilities ensures that they represent a valid probability distribution across all classes.

**Multinomial**

Multinomial Logistic Regression, also known as Softmax Regression, is an extension of logistic regression that is used for multi-class classification problems. It allows for the prediction of probabilities across multiple mutually exclusive classes.

In multinomial logistic regression, the goal is to model the relationship between the predictor variables and the probabilities of each class. Instead of modeling binary outcomes, it deals with multiple classes. The model estimates the probabilities of each class and assigns the observation to the class with the highest probability.

The mathematical formulation of multinomial logistic regression involves the use of the softmax function. Given a set of predictor variables and their corresponding weights, the softmax function calculates the probabilities for each class.

Mathematically, the softmax function is defined as:

$$
p_i= \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

where $p_i = P(Y=y_i\mid \boldsymbol{X}) = P(Y=y_i\mid X_1,X_2,\dots,X_N)$ is the 	 of class $i$. In OvR, we model $\eta$ as linear; here we model $z$ as linear. $z_i = \beta_{i,0}+\beta_{i,1}X_1+\beta_{i,2}X_2+\dots+\beta_{i,N}X_N$ is is the linear combination of the predictor variables (if we set intercept is True, then we have $\beta_{i,0}$) and their corresponding weights for class $i$, and the sum is taken over all classes $j$.

To fit a multinomial logistic regression model, the weights or coefficients are estimated using maximum likelihood estimation or other optimization algorithms. The objective is to find the values of the weights that maximize the likelihood of observing the training data given the model.

Note that softmax function naturally guarantees that the sum of probabilities of all classes is 1, so no normalization is needed. But OvR needs normalization for 3 or more classes (see the following for special case: binary classification).

**Special case: binary classification**

Note that for special case -- binary classification: in binary classification, ovr can guarantee that the sum of probabilities of 2 classes is 1 (it is to some extent intuitive. Mathematical proof should be easy).

Note that, different from multi-class classification, in binary classification (the target array only contains two different classes, like 0 and 1), `sklearn.linear_model.LogisticRegression` no longer gives the coefficients and intercepts for all classes, but only gives the coefficients and intercepts for class 1. But we can swap the class (0 → 1, 1 → 0) to get the coefficients and intercepts for the other class.

See code and comments in code for more details.

> `classes_: ndarray of shape (n_classes, )`
>
> A list of class labels known to the classifier.
>
> `coef_: ndarray of shape (1, n_features) or (n_classes, n_features)`
>
> Coefficient of the features in the decision function.
>
> coef_ is of shape (1, n_features) when the given problem is binary. In particular, when multi_class='multinomial', coef_ corresponds to outcome 1 (True) and -coef_ corresponds to outcome 0 (False).
>
> `intercept: _ndarray of shape (1,) or (n_classes,)`
>
> Intercept (a.k.a. bias) added to the decision function.
>
> If fit_intercept is set to False, the intercept is set to zero. intercept_ is of shape (1,) when the given problem is binary. In particular, when multi_class='multinomial', intercept_ corresponds to outcome 1 (True) and -intercept_ corresponds to outcome 0 (False).
>
> `n_features_in_: int`
>
> Number of features seen during fit.

Note: **'multinomial' (default option for multi-class) achieves better performance than 'ovr'.**

### Results

We digitize the output into $J$ intervals ($J$ classes for multi-class logistic regression).
$$
p_i= \frac{e^{\beta_{i,0}+\beta_{i,1}X_1+\beta_{i,2}X_2+\dots+\beta_{i,N}X_N}}{\sum\limits_{j=1}^J e^{\beta_{j,0}+\beta_{j,1}X_1+\beta_{j,2}X_2+\dots+\beta_{j,N}X_N}}, i=1,\ldots,J
$$
**Fitted Parameters** include a $J$ by $N$ matrix for coefficients and a $J$ length vector for intercepts.

Considering that logistic regression is a classification, I apply two classifications to enhance the performance: one for fitting (training) and the other for evaluation. 

For example, in the training stage, I classify data into class_num (e.g., class_num = 160) intervals (hitogram, by np.digitize); in test/evaluation stage, I evaluate the results with a same number of classes (e.g., reduced_class_num = 16). That is, for the example, class_num = 160 and reduced_class_num = 16, the in test/evaluation stage, classes 0, 1, ..., 15 become one class, i.e., 0; ...; classes 144, 145, ..., 159 become one class, i.e., 15.

Fixing the reduced class number, I enumerate the original class number to see what a original class number is better.

![](.\pics_CL090_230515\mse_correlation_r2_trend_curve_reduced_eval_reduced_class_num_16.png)

For this cell, Max Original Class Number is 480. The corresponding result is as follows.

**Results**:

|                         | Train data          | Test data          |
| ----------------------- | ------------------- | ------------------ |
| R2 score                | 0.49425421012452153 | 0.3095552619991193 |
| Mean squared error      | 1.7105263157894737  | 2.7222222222222223 |
| Correlation coefficient | 0.7442081479296433  | 0.6060794440849583 |

![Comparison (Logistic Linear Regression Reduced Evaluation 480 to 16, Test Set)](.\pics_CL090_230515\Comparison (Logistic Linear Regression Reduced Evaluation)\Comparison (Logistic Linear Regression Reduced Evaluation 480 to 16, Test Set).png)

![Comparison (Logistic Linear Regression Reduced Evaluation 480 to 16, Train Set)](.\pics_CL090_230515\Comparison (Logistic Linear Regression Reduced Evaluation)\Comparison (Logistic Linear Regression Reduced Evaluation 480 to 16, Train Set).png)

## Comparison of Different Regressions with Same Metric

Classify all the predictions into 16 intervals/classes for the evaluation.

### Ordinary Linear Regression

|                         | Train data         | Test data           |
| ----------------------- | ------------------ | ------------------- |
| R2 score                | 0.4011018872884826 | 0.36944077498899164 |
| Mean squared error      | 2.0255847953216373 | 2.486111111111111   |
| Correlation coefficient | 0.6340286499988343 | 0.6292717941149766  |

### Ridge Linear Regression

|                         | Train data         | Test data           |
| ----------------------- | ------------------ | ------------------- |
| R2 score                | 0.3974276657381052 | 0.36944077498899164 |
| Mean squared error      | 2.038011695906433  | 2.486111111111111   |
| Correlation coefficient | 0.6311655529544047 | 0.6290805964856068  |

### Elasticnet Linear Regression

|                         | Train data         | Test data          |
| ----------------------- | ------------------ | ------------------ |
| R2 score                | 0.3775436432301804 | 0.2954645530603258 |
| Mean squared error      | 2.1052631578947367 | 2.7777777777777777 |
| Correlation coefficient | 0.6144590454550366 | 0.566099913342907  |

### Power-Law Regression

|                         | Train data          | Test data          |
| ----------------------- | ------------------- | ------------------ |
| R2 score                | 0.42660530746169045 | 0.3377366798767063 |
| Mean squared error      | 1.939327485380117   | 2.611111111111111  |
| Correlation coefficient | 0.653816054551323   | 0.6064419746929176 |

### Exponential Regression

|                         | Train data         | Test data           |
| ----------------------- | ------------------ | ------------------- |
| R2 score                | 0.4291988756148981 | 0.32716864817261115 |
| Mean squared error      | 1.9305555555555556 | 2.6527777777777777  |
| Correlation coefficient | 0.6558085630583136 | 0.5985681611539189  |

### Logistic Regression

|                         | Train data          | Test data          |
| ----------------------- | ------------------- | ------------------ |
| R2 score                | 0.49425421012452153 | 0.3095552619991193 |
| Mean squared error      | 1.7105263157894737  | 2.7222222222222223 |
| Correlation coefficient | 0.7442081479296433  | 0.6060794440849583 |

## Delete Small Groups

Delete groups (axons) with less than 3 components.

### Exponential Regression

|                         | Train data          | Test data           |
| ----------------------- | ------------------- | ------------------- |
| R2 score                | 0.42530852338508673 | 0.36239542051959495 |
| Mean squared error      | 1.9437134502923976  | 2.513888888888889   |
| Correlation coefficient | 0.6527532920188056  | 0.6251978930658115  |

### Logistic Regression

|                         | Train data         | Test data          |
| ----------------------- | ------------------ | ------------------ |
| R2 score                | 0.4918767726507479 | 0.3095552619991193 |
| Mean squared error      | 1.7185672514619883 | 2.7222222222222223 |
| Correlation coefficient | 0.7430224939475898 | 0.6060794440849583 |

## Decay Correction

*Does decay influence dFF?*

For cell `CL090_230515` red data:

- mean(mean(rowdata1)) = 0.0664
- mean(mean(rowdata2)) = 0.0498
- mean(mean(rowdata3)) = 0.0252

For cell `CL090_230515` green data:

- mean(mean(rowdata1)) = 0.0378
- mean(mean(rowdata2)) = 0.0456
- mean(mean(rowdata3)) = 0.0377

For cell `CL075_230303` red data:

- mean(mean(rowdata1)) = 0.0278
- mean(mean(rowdata2)) = 0.0160
- mean(mean(rowdata3)) = 0.0139

For cell `CL075_230303` green data:

- mean(mean(rowdata1)) = -0.0066
- mean(mean(rowdata2)) = -0.0019
- mean(mean(rowdata3)) = -0.0081

