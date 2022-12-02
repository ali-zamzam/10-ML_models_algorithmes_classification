"""Bagging and boosting methods are "meta-algorithms" whose approach is to combine several machine learning algorithms 
into a predictive model, in order to reduce their variance or bias, and improve the final performance.

Both methods work in a similar way, and consist of 2 main steps:

1) Build different simple Machine Learning models on subsets of the original data.

2) Produce a new model from the assembly of the previous ones."""
# ------------------------------------------------------------------------------------------------------------
"""In the  exercise, we will use the 'letter-recognition.csv' data set which contains certain characteristics specific to 
images representing one of the 26 capital letters of the Latin alphabet, as well as the 'letter' column containing the 
letter in question.
"""
"""(a) Load the AdaBoostClassifier package from the sklearn.ensemble package.
(b) Load the DecisionTreeClassifier package from the sklearn.tree library.
(c) Load the train_test_split package from the sklearn.model_selection library.
(d) Load the f1_score package from the sklearn.metrics library.
(e) Load the pandas library as pd.
(f) Load the numpy library as np."""

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/letter-recognition.csv")

df.head()

"""Separate the features into a DataFrame named data, and the target variable into target."""
data = df.drop("letter", axis=1)
target = df["letter"]

# ------------------------------------------------------------------------------------------------

"""Separate the datasets into a training set and a test set. The size of the test set should be 30% of 
the total amount of data available."""


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

# ------------------------------------------------------------------------------------------------

"""(a) Create a dtc classification model, with parameter max_depth=5, using the DecisionTreeClassifier method.
(b) Train the algorithm on the training set (X_train and y_train)."""

dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------
"""Evaluation of the first classification model"""


"""Calculate the rate of correct predictions of the classification model on the test set using the score 
method."""

dtc.score(X_test, y_test)

"""Perform model predictions on X_test, and display the corresponding confusion matrix.
"""
y_pred = dtc.predict(X_test)
pd.crosstab(y_test, y_pred, rownames=["Real Class"], colnames=["Predicted Class"])

# ------------------------------------------------------------------------------------------------
"""Boosting Algorithm"""

"""Scikit-learn allows the application of Adaboost as a Boosting algorithm.
The objective of this part is to create a new classification model from sequences of decision trees as dtc defined 
above.
"""
"""(a) Create again a classifier ac, having as parameters: base_estimator=dtc, and n_estimators=400, using the 
AdaBoostClassifier class from the ensemble package.

(b) Train the algorithm on the set (X_train, y_train).
(c) Calculate the accuracy (good prediction rate) of the new classification model on the test sample."""

ac = AdaBoostClassifier(base_estimator=dtc, n_estimators=400)
ac.fit(X_train, y_train)


ac.score(X_test, y_test)

# ------------------------------------------------------------------------------------------------
"""(d) Display the confusion matrix of the test set."""
y_pred = ac.predict(X_test)
pd.crosstab(y_test, y_pred)

# ------------------------------------------------------------------------------------------------
"""Bagging or Bootstrap AGGregatING"""

"""(a) Import BaggingClassifier from sklearn.ensemble.
(b) Create a new classifier bc, having as parameters: n_estimators=1000, and oob_score=True to calculate the Out Of 
Bag error.
(c) Train the algorithm on the training set (X_train and y_train).
(d) Display the model's Out Of Bag error using the oob_score_ attribute."""

from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier(n_estimators=1000, oob_score=True)
bc.fit(X_train, y_train)
bc.oob_score_


"""(e) Calculate the accuracy (good prediction rate) of the new classification model on the test sample."""

bc.score(X_test, y_test)

"""Display the test set confusion matrix."""
y_pred = bc.predict(X_test)
pd.crosstab(y_test, y_pred)
