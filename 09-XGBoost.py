"""XGBoost only works with numeric vectors: all categorical variables will need to be converted into numeric variables.
The simplest solution for this is 'One-hot' encoding, or 'dummification', which consists of creating for 
each variable as many dichotomous variables as there are modalities present.

One of the peculiarities of XGBoost is that this algorithm requires the data to be in matrix format 
(so it does not accept DataFrames for example).

The xgboost package allows you to create an xgb.DMatrix object containing the variable matrix and the prediction 
vector, to be inserted respectively in the data and label arguments."""
# ------------------------------------------------------------------------------------------------
"""Adjustable parameters in XGBoost

XGBoost contains its own train() function which allows you to train a model, specifying the training set and the 
various parameters to be set.

XGBoost contains a large number of hyperparameters that can be modified and tuned to increase accuracy.
Each parameter has a significant role to play in the performance of the model.

xgboost settings can be separated into three categories:

The general parameters, which control among other things:
booster: The type of booster used (by default gbtree).
nthread: The number of cores to use for parallel computing (by default all available cores are used).
The booster parameters (we will limit ourselves here to the case of trees):
num_boost_round: The maximum number of iterations/trees built (100 by default).
learning_rate: Controls the 'learning rate'. At each boosting step, a constant is introduced into the model update 
formula. It reduces the weight achieved in relation to performance to prevent overfitting. A low value results in a 
more robust model to overfitting, but slower computation and convergence. Consider increasing the number of trees when 
learning_rate is low (is 0.3 by default, and must be between 0 and 1).
min_split_loss: Minimum loss reduction required to perform an additional split on a tree node. The larger it is, the 
more conservative the algorithm will be.
max_depth: Controls the depth of trees. The deeper the trees, the more complex the model and the greater the chances 
of overfitting (6 by default).
min_child_weight: Stopping criterion relating to the minimum size of the number of observations in a node (1 by default).
subsample: Allows to use a subsample of the training dataset for each tree (is 1 by default, no subsampling; and must 
be between 0 and 1).
colsample_bytree: Allows the use of a certain number of variables among those of origin (is worth 1 by default, all 
the variables are selected; and must be included between 0 and 1).
reg_lambda and reg_alpha: respectively control the L1 and L2 regularization on the weights (equivalent to Ridge and 
Lasso regression).
Learning parameters:
objective: Objective function to use:
binary:logistic for binary classification. Returns the probabilities for each class.
reg:linear for regression.
multi:softmax for multiple classification using the softmax function. Returns the predicted labels.
multi:softprob for multiple classification using the softmax function. Returns the probabilities for each class.
eval_metric: Evaluation metric (by default prediction error for classification, RMSE for regression).
The available metrics are: mae(Mean Absolute Error), Logloss, AUC, RMSE, error mologloss, etc...
early_stopping_rounds: to stop training when the evaluation on the test set does not improve any more during a certain 
number of iterations. The validation error must decrease at least every early_stopping_rounds to continue training.
Beyond regression and classification, XGBoost supports all user-defined objective functions, as well as custom 
evaluation metrics. This gives it great flexibility.

The dtrain parameter lets you specify the training matrix to use, and evals takes a list as an argument that allows 
it to display the scores obtained for the samples it contains."""
# ------------------------------------------------------------------------------------------------
"""With a learning_rate=1, we notice that very quickly, the error of the validation sample stops decreasing, and the 
gap widens between the two errors as the iterations progress.

At each iteration of the Gradient Boosting algorithm, new trees are created to correct residual errors in the 
predictions from the existing sequence of trees.

So the model can adapt very quickly and then overfit on the training dataset.

One technique to slow down learning in the Gradient Boosting algorithm is to apply a weighting factor to the 
corrections made by new trees as they are added to the model.

This weighting is most often called the learning rate.
By adding this shrinkage factor (i.e. a learning_rate < 1), adding each tree will have less impact on the model, 
therefore more trees need to be added to the model.

In addition, the lower the learning_rate, the better the model converges to its optimum, but the longer this 
convergence takes, therefore the slower the training of the model.
Conversely, a high learning_rate allows faster, but less optimal training.

To take full advantage of XGBoost, it is therefore necessary to favor the lowest possible learning_rate and a high 
number of trees, while keeping a reasonable calculation time."""
# --------------------------------------------------------------------------------------------------------------
"""To avoid wasting precious time, it is possible to use the early_stopping_rounds parameter to stop learning when the 
validation error no longer decreases beyond a certain number of iterations.

If early_stopping_rounds is used, the model will have 3 additional attributes best_score, best_iteration and 
best_ntree_limit.
By default, the model returned will be that of the last iteration. To use the best to make predictions, it is possible 
to use for example xgb_model.predict(test, ntree_limit= xgb_model.best_ntree_limit).

The xgb.plot_importance function displays a bar graph displaying in descending order the importance of each feature 
for a model."""

# --------------------------------------------------------------------------------------------------------------
"""We can see that there are three options to measure the importance of features in XGBoost:

'Weight' (default): the percentage representing the relative number of times a feature appears in the model trees.
'Cover': The number of times a feature is used to split data in the set of trees, weighted by the number of 
training data that passes through these splits.
'Gain': The average reduction of the loss function obtained when using a feature to separate a branch.
It is important to understand how these metrics work to properly interpret the results displayed.
For example, using 'Weight', it makes sense that a variable with a large number of possible values and which 
can be used a large number of times in a tree (like age) would have much greater importance than a variable 
binary that can only be used at most once in each tree (like the genus).
However, gender could be a feature that greatly affects the final results and has great importance, based on 'Gain' or 'Cover'.

A higher value of the 'Gain' metric for one feature compared to another implies that it is more important for 
generating a prediction.
It is therefore, for many, the most relevant metric for interpreting the relative importance of each feature."""
# --------------------------------------------------------------------------------------------------------------
"""XGBoost contains its own xgb.cv() function which allows to use cross-validation for the evaluation of a model, 
without the need to go through an external package.
It is used in the same way as train(), with additional arguments like nfold to choose the number of samples for 
cross-validation, and returns an array of training and testing scores for each iteration."""
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
"""In the  exercise we will work with the 'adult_income.csv' dataset which includes nearly 50,000 observations 
and 15 attributes that are both categorical and continuous.
It contains information from the 1994 US Census for 4,983 US citizens in 1994.
The last column 'income' is a binary variable taking the value '<=50k' if the individual earns less than 50000, 
nd '>50k' if he earns more.
The 'income' variable will be our target variable throughout this exercise.
"""
# --------------------------------------------------------------------------------------------------------------
"""(a) Import numpy and pandas packages.
(b) Import the train_test_split function from sklearn.model_selection.
(c) Import xgboost under xgb.
(d) Read the adult.csv file into an adult DataFrame."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

adult = pd.read_csv("data/adult_income.csv")
adult.head()

"""(e) Replace '?' values variables occupation, workclass and native.country by NaN (np.nan)."""
adult = adult.replace("?", np.nan)

"""
(f) Remove the 4th column from the data set, corresponding to the education variable.
(g) Replace the countries 'Cambodia', 'China', 'Hong', 'India', 'Iran', 'Japan', 'Laos', 'Philippines', 'Taiwan', 
'Thailand', 'Vietnam' by 'Asia'.
(h) Replace the countries Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'Guatemala', 'Haiti', 
'Honduras', 'Jamaica', 'Mexico City', ' Nicaragua','Peru', 'Puerto-Rico', 'Trinadad&Tobago', 'South' by 
'Center & South America'.
(i) Replace countries 'England', 'France', 'Germany', 'Greece', 'Holand-Netherlands', 'Hungary', 'Ireland', 
'Italy', 'Poland', 'Portugal', 'Scotland' , 'Yugoslavia' by 'Europe'.
(j) Replace the countries 'United-States' and 'Canada' with 'Canada&USA'."""


adult = adult.drop("education", axis=1)

adult.replace(
    [
        "Cambodia",
        "China",
        "Hong",
        "India",
        "Iran",
        "Japan",
        "Laos",
        "Philippines",
        "Taiwan",
        "Thailand",
        "Vietnam",
    ],
    "Asia",
    inplace=True,
)

adult.replace(
    [
        "Columbia",
        "Cuba",
        "Dominican-Republic",
        "Ecuador",
        "El-Salvador",
        "Guatemala",
        "Haiti",
        "Honduras",
        "Jamaica",
        "Mexico",
        "Nicaragua",
        "Peru",
        "Puerto-Rico",
        "Trinadad&Tobago",
        "South",
    ],
    "Center & South America",
    inplace=True,
)

adult.replace(
    [
        "England",
        "France",
        "Germany",
        "Greece",
        "Holand-Netherlands",
        "Hungary",
        "Ireland",
        "Italy",
        "Poland",
        "Portugal",
        "Scotland",
        "Yugoslavia",
    ],
    "Europe",
    inplace=True,
)

adult.replace(["United-States", "Canada"], "Canada&USA", inplace=True)
# --------------------------------------------------------------------------------------------------------------
"""(k) Split adult into two DataFrames, features and target, containing respectively the explanatory variables and 
the income variable to be predicted.
(l) Transform the labels '>50K' and '<=50K' of target into 1 and 0 respectively.
(m) Create a features_matrix where all categorical variables are dichotomized."""

features = adult.drop("income", axis=1)
target = adult["income"]

target = [1 if x == ">50K" else 0 for x in target]

features_matrix = pd.get_dummies(features)

"""(n) Create from features_matrix and target, a validation set representing 10% of the data, then a training set 
containing 80% of the remaining data (X_train, y_train) and a test set (X_test, y_test)."""

X, X_valid, y, y_valid = train_test_split(features_matrix, target, test_size=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --------------------------------------------------------------------------------------------------------------
"""(o) Create a DMatrix object called train from X_train and y_train.
(p) Create the equivalent test and valid DMatrix objects."""

train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)
valid = xgb.DMatrix(data=X_valid, label=y_valid)


"""(a) Create a params dictionary containing the default parameters, with booster: "gbtree, learning_rate: 1, objective: "
binary:logistic.
(b) Using the xgb.train() function, train a model named xgb1 on train, with the following parameters params and
 arguments: num_boost_round = 100, early_stopping_rounds= 15, evals= [(train, 'train' ), (test, 'eval')]).
The function will show the evolution of the error for the two data sets over the boosting iterations."""

params = {"booster": "gbtree", "learning_rate": 1, "objective": "binary:logistic"}

xgb1 = xgb.train(
    params=params,
    dtrain=train,
    num_boost_round=100,
    evals=[(train, "train"), (test, "eval")],
)
# --------------------------------------------------------------------------------------------------------------
"""c) Create a new xgb2 model, identical to the previous one, with a learning_rate of 0.01, and 700 boosting iterations."""

params = {"booster": "gbtree", "learning_rate": 0.01, "objective": "binary:logistic"}

xgb2 = xgb.train(
    params=params,
    dtrain=train,
    num_boost_round=700,
    evals=[(train, "train"), (test, "eval")],
)
# --------------------------------------------------------------------------------------------------------------
"""(d) Display the importance graph for xgb2, limiting the number of displayed variables to 15 thanks to the 
max_num_features parameter"""

xgb.plot_importance(xgb2, max_num_features=15)

"""(e)display the important features according to the different metrics available."""

types = ["weight", "gain", "cover", "total_gain", "total_cover"]

for f in types:
    xgb.plot_importance(
        xgb2, max_num_features=15, importance_type=f, title="importance: " + f
    )

# --------------------------------------------------------------------------------------------------------------
"""(f) Return, in bst_cv, the results obtained by 3-sample cross-validation on train, with 100 boosting iterations 
per step, with early_stopping_rounds= 60 to stop the training of each sample if the evaluation does not improve 
for 60 iterations.
(g) Display bst_cv."""


bst_cv = xgb.cv(
    params=params, dtrain=train, num_boost_round=100, nfold=3, early_stopping_rounds=60
)
bst_cv

# --------------------------------------------------------------------------------------------------------------
"""(h) Store in preds the probabilities obtained with xgb2 on test.
(i) Create a Series xgbpreds containing the labels corresponding to the probabilities obtained, using a 
threshold of 0.5 (i.e. 1 if the probability >=0.5, 0 otherwise).
(j) Display a confusion matrix between xgbpreds and test set labels."""

preds = xgb2.predict(test)

xgbpreds = pd.Series(np.where(preds > 0.5, 1, 0))

pd.crosstab(xgbpreds, pd.Series(y_test))

# --------------------------------------------------------------------------------------------------------------
"""It is possible to obtain the prediction error of the model on a sample directly from a DMatrix, thanks to 
the eval() method.
"""
"""(k) Display the model error on the validation sample contained in valid."""

xgb2.eval(valid)
