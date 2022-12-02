"""Voting Classifier and Stacking"""

"""the objective of these methods is to assemble a collection of strong and already efficient estimators in order to take 
advantage of the particular advantages of each."""

# -----------------------------------------------------------------------------------------------------
"""The dataset used in the following concerns the diabetes syndrome, and the objective is to infer people likely to have 
diabetes from diagnosed measures that can influence the onset of diabetes. The data we are going to process contains 
eight explanatory variables and one target variable:
"""
# Field                         Description
# ----------------------------------------------------------------
# Pregnancies                   The number of pregnancies
# Glucose                       Plasma glucose concentration at 2 hours in an oral glucose tolerance test
# BloodPressure                 Diastolic Blood Pressure ( mm Hgmm Hg )
# SkinThickness                 Triceps skinfold thickness ( mmmm )
# Insulin                       2 hour serum insulin ( mu U / mlmu U / ml )
# BMI                           Body Mass Index (weight in kg / (height in m)2weight in kg / (height in m)2 )
# DiabetesPedigreeFunction      Diabetes pedigree function
# Age                           Age (years)
# Outcome                       Class variable (0 or 1)

"""The main objective will be to use the Ensemble methods above to be able to classify a set of North American Indian 
individuals into two groups: diabetic and non-diabetic."""
# -----------------------------------------------------------------------------------------------------
"""(a) Load the VotingClassifier and StackingClassifier classes, from sklearn.ensemble.
(b) Load train_test_split, cross_validate and KFold functions from sklearn.model_selection.
(c) Load f1_score function from sklearn.metrics library.
(d) Load LogisticRegression class from sklearn.linear_model.
(e) Load the RandomForestClassifier class from sklearn.ensemble.
(f) Load the KNeighborsClassifier package from sklearn.neighbors.
(g) Load the pandas library as pd.
(h) Read the diabetes.csv file into a DataFrame called df and display the first few lines."""

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/diabetes.csv")

df.head(3)

"""The explanatory variables or features of each individual are represented by the first eight attributes of the dataset, 
and the variable to be predicted is Outcome."""

# -----------------------------------------------------------------------------------------------------
"""(i) Create a DataFrame data which will contain the features.
(j) Store the target variable in the target variable."""

data = df.drop("Outcome", axis=1)
target = df["Outcome"]

# or
# data = df.iloc[:,:-1]
# target = df.Outcome
# -----------------------------------------------------------------------------------------------------
""" Divide the arrays into a training set and a test set.
Specifically, data will be split into X_train and X_test and target will be split into y_train and 
y_test."""

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=4
)

# -----------------------------------------------------------------------------------------------------
"""Voting Classifier """

"""The Voting Classifier is a meta-classifier that combines several models, sometimes similar but often 
conceptually different by majority voting."""

"""(a) Create the following three classifiers named respectively clf1, clf2 and clf3:
A KNeighborsClassifier class model, with 3 neighbors (n_neighbors=3).
A RandomForestClassifier model, with the random_state=123 parameter.
A LogisticRegression model, with the max_iter=1000 parameter to ensure model convergence."""

clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = RandomForestClassifier(random_state=123)
clf3 = LogisticRegression(max_iter=1000)
# -----------------------------------------------------------------------------------------------------
"""The construction of a Voting Classifier is done using an instance of the VotingClassifier class which 
takes as parameters:

estimators: a list containing for each estimator a label and the name of the estimator.
voting: allows you to specify the voting method (hard or soft).
As a reminder :
with the hard vote: the label of the final class is that of the class most frequently predicted by the 
classification models.
With soft voting: the final class label is obtained by averaging the class probabilities.
"""

"""(b) Using the documentation, create vclf, an instance of the VotingClassifier class which takes the 
three previously created models as parameters, with the hard voting mode."""

vclf = VotingClassifier(
    estimators=[("knn", clf1), ("rf", clf2), ("lr", clf3)], voting="hard"
)
# -----------------------------------------------------------------------------------------------------
"""(c) Using the Kfold function, create cv3: a 3-part cross-validator (folds), with parameters 
random_state=111 and shuffle=True.
(d) Using the cross_validate() function, display for each of the 3 individual classifiers, as well as for 
the Voting Classifier:
the mean and the standard deviation of the score of good prediction rate ('accuracy').
the mean and standard deviation of the f1-score ('f1') obtained by cross-validation 
with cv3 on (X_train, y_train)."""

cv3 = KFold(n_splits=3, random_state=111, shuffle=True)

for clf, label in zip(
    [clf1, clf2, clf3, vclf],
    ["KNN", "Random Forest", "Logistic Regression", "Voting Classifier"],
):
    scores = cross_validate(clf, X_train, y_train, cv=cv3, scoring=["accuracy", "f1"])
    print(
        "[%s]: \n Accuracy: %0.2f (+/- %0.2f)"
        % (label, scores["test_accuracy"].mean(), scores["test_accuracy"].std()),
        "F1 score: %0.2f (+/- %0.2f)"
        % (scores["test_f1"].mean(), scores["test_f1"].std()),
    )
# -----------------------------------------------------------------------------------------------------
# Like any scikit-learn model, it is also possible to apply the GridSearchCV() function to the meta-classifier,
# to select the hyper-parameters of each sub-model.
# It is also possible to perform a grid search on all the arguments of the estimators parameter, to select the
# models giving the best results.

# Here is an example of use:

# from sklearn.model_selection import GridSearchCV

# params = {'knn__n_neighbors': [5, 9],
#            'rf__n_estimators': [20, 100, 200],
#            'svm__C': [0.01, 0.1, 1]
#             'estimators': [[('knn', clf1), ('lr', clf3)], [('knn', clf1), ('rf', clf2), ('svm', clf4)]]
#            }

# grid = GridSearchCV(estimator=vclf, param_grid=params, cv=5)
# grid = grid. fit(X_train, y_train)
# print(grid.best_params_)
# -----------------------------------------------------------------------------------------------------
"""Stacking"""

"""Stacking is a method that uses the outputs of different classifiers as inputs to a new meta-classifier, 
defined upstream.
The Stacking model with scikit-learn is created from the StackingClassifier class.
It is used in the same way as the VotingClassifier, but an additional parameter, final_estimator, makes it 
possible to indicate the model to be used as meta-classifier.
"""
"""(a) Create sclf, an instance of the StackinClassifier class which takes as parameters the three models clf1, 
clf2 and clf3, and as final estimator also the logistic regression, clf3.
(b) Display the results obtained by the Stacking Classifier, under the same cross-validation conditions as before."""


sclf = StackingClassifier(
    estimators=[("knn", clf1), ("rf", clf2), ("lr", clf3)], final_estimator=clf3
)

scores = cross_validate(sclf, X_train, y_train, cv=cv3, scoring=["accuracy", "f1"])

print(
    "[StackingClassifier]: \n Accuracy: %0.2f (+/- %0.2f)\n"
    % (scores["test_accuracy"].mean(), scores["test_accuracy"].std()),
    "F1 score: %0.2f (+/- %0.2f)" % (scores["test_f1"].mean(), scores["test_f1"].std()),
)

"""The cross-validation results are good and at least equal to the performance of the best individual classifier."""
# -----------------------------------------------------------------------------------------------------

"""Now let's evaluate our two Ensemble models on the test set.

(c) Train vclf and sclf successively on the training set, (X_train, y_train).
(d) Display the correct prediction rate scores obtained by the two models on (X_test, y_test)."""

vclf.fit(X_train, y_train)
sclf.fit(X_train, y_train)

print("Acc :", vclf.score(X_test, y_test))
print("Acc :", sclf.score(X_test, y_test))

"""(e) Train the best model (clf2) on the training set, and display the obtained score on the test set.
"""
clf2.fit(X_train, y_train)

print("Acc :", clf2.score(X_test, y_test))
