"""Classification of wines using the Support Vector Machine (SVM)"""

"""The objective of this exercise is to create a multi-class classification model from the data in the wine dataset.
This dataset compiles the results of a chemical analysis of wines from the same region of Italy, but from 3 different 
vineyards. 
The analysis determines the amount of 13 constituents found in each of the three types of wine."""

"""((((SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the 
data are not otherwise linearly separable. A separator between the categories is found, then the data are transformed in such 
a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict 
the group to which a new record should belong.

- The important arguments of the svm.SVC function are: the inverse of the C regularization parameter, 
the kernel function to use,(mainly linear, rbf or poly), and (gamma) the coefficient for (rbf and poly kernels).)))))"""
# --------------------------------------------------------------------------------------------------------------------
"""(a) Load pandas and numpy libraries as pd and np.
(b) Load the svm package from the sklearn library.
(c) Load the model_selection package from the sklearn library.
(d) Load the train_test_split package from the sklearn.model_selection library.
(e) Load the preprocessing package from the sklearn library.
(f) Read the wine.csv file into a DataFrame called wine."""

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, svm
from sklearn.model_selection import train_test_split

wine = pd.read_csv("data/wine.csv")

print(wine.head())


"""(g) Display the information of the DataFrame."""

print(wine.info())
# --------------------------------------------------------------------------------------------------------------------
"""
(i) Create a variable data whose features you store.
(j) Store target data in target variable.
"""

data = wine.iloc[:, 1:]

target = wine.iloc[:, 0]


"""(k) Divide the matrices defined above into a training and testing set. Specifically, data will be split 
into X_train and X_test and target will be split into y_train and y_test."""

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
# --------------------------------------------------------------------------------------------------------------------
"""(n) Create a scaler object and apply it to X_train to return the centered-scaled X_train_scaled array."""

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

"""(o) Display the mean and standard deviation of the columns of X_train_scaled."""

print(X_train_scaled.mean(axis=0))

print(X_train_scaled.std(axis=0))


"""(p) Apply the same transformation to X_test in the X_test_scaled array.
(q) Display the mean and standard deviation of the columns of X_test_scaled."""

X_test_scaled = scaler.transform(X_test)

print(X_test_scaled.mean(axis=0))

print(X_test_scaled.std(axis=0))

# --------------------------------------------------------------------------------------------------------------------
"""Classification by support vector machines"""

"""A classification model by SVM is created on Scikit-learn using the svm.SVC function 
(SVR in the case of a regression)."""

"""
(a) Create a clf classifier, with parameters gamma=0.01 and kernel='poly', using the SVC method of the svm package.
(b) Train the algorithm on the training set (X_train_scaled and y_train)."""

clf = svm.SVC(gamma=0.01, kernel="poly")

clf.fit(X_train_scaled, y_train)

# --------------------------------------------------------------------------------------------------------------------
"""Evaluation of the classification model"""

"""It is then possible to calculate the prediction for the data present in the test set, and construct the confusion 
matrix.

(a) Perform the predictions on the test set and store them in the y_pred variable.
(b) Display a confusion matrix from these predictions."""

y_pred = clf.predict(X_test_scaled)
pd.crosstab(y_test, y_pred, rownames=["Real Class"], colnames=["Predicted Class"])


"""(c) Create a parameter dictionary containing the possible values taken for parameter 
C:[0.1,1,10], for kernel: ['rbf', 'linear','poly'] and for gamma:[0.001, 0.1, 0.5]."""


parametres = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear", "poly"],
    "gamma": [0.001, 0.1, 0.5],
}

"""(d) Apply the model_selection.GridSearchCV() function to the clf model, specifying in the param_grid argument 
the parameter grid created above. Return the classifier thus created in grid_clf.
The scoring argument allows you to choose the metric you want to use to evaluate the performance of the models, 
by default it is accuracy."""

grid_clf = model_selection.GridSearchCV(estimator=clf, param_grid=parametres)

"""
(e) Train grid_clf on the training set, (X_train_scaled, y_train). Save results to grid object."""

grid = grid_clf.fit(X_train_scaled, y_train)

""" (cv_results_) returns a dictionary of all the evaluation metrics from the gridsearch. 
( from_dict() ) function is used to construct DataFrame from dict of array-like or dicts

(f) display all possible combinations of hyperparameters and the average performance of the associated 
model by cross-validation."""

print(pd.DataFrame.from_dict(grid.cv_results_).loc[:, ["params", "mean_test_score"]])

# The results already seem much better with the parameter 'gamma=0.1 or 0.5' for the 'poly' kernels
# than the parameter gamma= 0.01.

"""
- The best_params_ attribute of the created model is used to display the parameters that gave the best score and 
retained by default.

(g) Show the best grid settings for our grid_clf model."""

print(grid_clf.best_params_)

# --------------------------------------------------------------------------------------------------------------------
"""(h) perform class predictions using the grid_clf model on the test set and store them in the y_pred variable.
(i) Display a confusion matrix from these predictions."""

y_pred = grid_clf.predict(X_test_scaled)
pd.crosstab(y_test, y_pred, rownames=["Real Class"], colnames=["Predict Class"])

# --------------------------------------------------------------------------------------------------------------------
"""The next cell displays the model's learning curve, i.e. the different scores (here: accuracy) obtained on the 
training sample and measured by cross-validation, depending on the sample size learning chosen"""

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

%matplotlib inline 

train_sizes, train_scores, valid_scores = learning_curve(svm.SVC(kernel='linear', C= 1), data, target, train_sizes=[50, 80, 110, 140], cv=5)

plt.xlabel("Training examples")
plt.ylabel("Score")

train_sizes=[50, 70, 80, 100, 110, 118]

train_sizes, train_scores, test_scores = learning_curve(
    grid_clf, data, target, n_jobs=4, train_sizes=train_sizes)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
