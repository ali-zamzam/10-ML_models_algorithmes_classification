"""
In this exercise, we will study an example of using logistic regression in Python.
The packages used will be pandas, scikit-learn and its sub-packages including linear_model, metrics, preprocessing and 
model_selection.

This exercise focuses on the parameters that can influence the admission of a candidate to high school.
The data we are going to process contains four variables:

admit: binary variable, which indicates whether a candidate is admitted (admit = 1) or not (admit = 0).
gre(Graduate Record Examination): an English test created and managed by ETS.
gpa(Grade Point Average): the average of a student's grades.
rank: rank of the candidate, 1 being the best score and 4 the lowest."""
# ------------------------------------------------------------------------------------------------------------------------
"""Data preparation and modeling"""

"""
(a) Import linear_model and preprocessing submodules from sklearn library.
(b) Import the train_test_split function from the sklearn.model_selection submodule.
(c) Import pandas and numpy libraries under pd and np aliases.
(d) Read the 'admissions.csv' file into a DataFrame named df.
(e) Display the first lines of df to verify that the import was successful."""

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/admission.csv")

print(df.head())

# ------------------------------------------------------------------------------------------------
"""(f) Display information from DataFrame df."""
print(df.info())

print(df.dropna())
"""Display the average of the gre variable by group of admitted/unadmitted students, thanks to the combination of the 
groupby and mean methods."""

print(df[["gre", "admit"]].groupby("admit").mean())
# ------------------------------------------------------------------------------------------------

"""
Using pandas cut function, discretize the gre variable from df. We will consider the following classes:
- If the score is between 200 and 450, then the score is 'bad'.
- If the score is between 450 and 550, then the score is 'average'.
- If the score is between 550 and 620, then the score is 'average+'.
- If the score is between 620 and 800 (maximum score), then the score is 'good'.

(j) Cross-reference the admitted variable of df with our discretization using pandas crosstab function.
(k) What is the relationship between a student's GRE test score and admission?"""
# (The number of labels must therefore be equal (bins-1 )--> bins 5 values and label 4 values)

test_gre = pd.cut(
    x=df["gre"],
    bins=[200, 450, 550, 620, 800],
    labels=["bad", "average", "average +", "good"],
)

print(pd.crosstab(df["admit"], test_gre))

"""The higher the GRE score, the more likely the candidate seems to be admitted."""

# ------------------------------------------------------------------------------------------------
"""(l) Discretize the variable gpa of df. We will consider the following classes:
If the average is between 2 and 2.8, then the student's level is 'bad'.
If the average is between 2.8 and 3.2, then the student's level is 'average'.
If the average is between 3.2 and 3.6, then the student's level is 'average+'.
If the average is between 3.6 and 4, then the student's level is 'good'.
(m) Cross the admitted variable of df with our discretization. We will use the normalize argument 
of the crosstab function to obtain the proportions of admissions according to the level of the 
student."""

grade_levels = pd.cut(
    x=df["gpa"],
    bins=[2, 2.8, 3.2, 3.6, 4],
    labels=["bad", "average", "average+", "good"],
)

pd.crosstab(df["admit"], grade_levels, normalize="columns")  # or normalize = 1

# ------------------------------------------------------------------------------------------------
"""Dichotomization is necessary for machine learning models of linear types that we will see in the following. 
Indeed, these models are incapable of interpreting qualitative variables. Thanks to the dichotomization we have 
transformed this qualitative variable into a "quantitative" variable that can be interpreted by a machine learning model."""


"""
- (o) Using pandas get_dummies function, dichotomize the discretizations done in the previous questions. 
We will add to the new columns the prefixes 'level' and 'gre'.
- (p) Using join or concatenation (merge, concat), merge the new DataFrames obtained with df."""

df = df.join(pd.get_dummies(grade_levels, prefix="grade_lev"))
df = df.join(pd.get_dummies(test_gre, prefix="gre"))


"""(q) Also apply the dichotomization to the 'rank' variable of df, and join the obtained DataFrame to df."""

# df = df.join(pd.get_dummies(df.rank, prefix='rank')) ????????

rank = pd.cut(
    x=df["rank"], bins=[1, 2, 2.5, 3, 4], labels=["good", "average+", "average", "bad"]
)

print(pd.crosstab(df["admit"], rank))


df = df.join(pd.get_dummies(rank, prefix="rank"))

"""(r) Display information of new DataFrame df."""
print(df.info())

df.head()
# ------------------------------------------------------------------------------------------------

"""(s) Create a DataFrame data in which you will store the dichotomized features starting with grade_level and gre 
as well as rank_good. We choose to use only these variables to train our model.
(t) Assign the target data to the target variable."""

data = df.iloc[:, 4:13]

target = df["admit"]

"""u) Display column names of data."""

print(data.columns)

# -------------------------------------------------------------------------------------------------------------------
"""Test the performance of the classification Model(Classification by logistic regression)"""

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=66
)

"""In the case of classification using logistic regression in Python, the optimization algorithms used are: 
L_BFGS (Limit_memory BFGS), SAG (Stochastic Average Gradient), Newton_cg (Newton Conjugate Gradient)."""

"""To build a classification model, we need to train our model on the training set only.

By specifying the inverse of the C regularization parameter (hyperparameters)(there are other parameters you can give, 
such as the optimization algorithm):

(b) Create a clf Classifier, with parameter C = 1.0, using the LogisticRegression method of the linear_model package.
(c) Train the algorithm on the training set (X_train and y_train).
"""
clf = linear_model.LogisticRegression(C=1.0)


#  or clf = LogisticRegression() (but we need to import LogisticRegression from sklearn.linear_model)
clf.fit(X_train, y_train)
# -------------------------------------------------------------------------------------------------------------------
"""Evaluation of the classification model"""

"""
- It is then possible to calculate the prediction for the data present in the test set, and construct the confusion matrix.

(a) Predict the test set data and store it in the y_pred variable.
(b) Create and display the cm confusion matrix."""

y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_test, y_pred)
print(conf)

# output:
# [[58  2]
#  [18  2]]

## Method 2: using pandas
# cm = pd.crosstab(y_test, y_pred, rownames=['Actual class'], colnames=['Predicted class'])
# cm

"""(c) Calculate the rate of good predictions of the model."""
clf.score(X_test, y_test)

# or
# (58 + 2) / (58 +2 + 18 + 2)

# -------------------------------------------------------------------------------------------------------------------

"""The classification_report() function of the sklearn.metrics sub-module allows you to display some of 
these additional metrics, with as arguments the vector of the true labels, and that of the predicted labels.

(d) View the classification report of our predictions."""
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# -------------------------------------------------------------------------------------------------------------------

"""By default, the predict() method of a classifier in the case of a logistic regression classifies individuals as 
positive when the probability of belonging to the positive class is greater than the threshold of 0.5, and negative 
otherwise.

Sometimes, you may want to modify this classification threshold.
The predict_proba method returns, for given individuals, not the class predictions but the probabilities of 
belonging to each of the two classes.

Thus, for example, it is possible to classify individuals according to the desired threshold.

(e) Create an array probs containing the probabilities for the individuals of X_test to belong to class 0 or class 1.
(f) Create a vector y_preds which, for each line of probs is 1 if the probability of belonging to class 1 is greater 
than 0.4, and 0 otherwise.
(g) Display a confusion matrix between the true labels of y_test and y_preds.
"""

probs = clf.predict_proba(X_test)

y_preds = np.where(probs[:, 1] > 0.4, 1, 0)

cm = pd.crosstab(y_test, y_preds, rownames=["real Class"], colnames=["predict Class"])
print(cm)

# -------------------------------------------------------------------------------------------------------------------
"""(h) Import the roc_curve() and auc() functions.
(i) Apply the roc_curve() function to y_test and the second column of probs, specifying that the positive label 
in our case is 1. Store the returned results in the arrays fpr, tpr, thresholds.
(j) Calculate in roc_auc the AUC corresponding to the values of fpr and tpr."""

from sklearn.metrics import auc, roc_curve

fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

"""(k) From the fpr, tpr and roc_auc variables, create a reproduction of a graph """
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, color="orange", lw=2, label="Model clf (auc = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random (auc = 0.5)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False positive rates")
plt.ylabel("True positive rates")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
