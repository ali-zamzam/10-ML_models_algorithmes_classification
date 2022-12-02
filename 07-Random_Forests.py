"""Random Forests"""

"""
Random forests have multiple advantages over other classification models:

They are multi-class classification models, efficient on high-dimensional data.
These are robust statistical methods for identifying outliers.
In general, they avoid overfitting, and do not require cross-validation thanks to "Out of bag" samples.
However, Random Forest models often have a longer learning time than classical models, and are difficult to interpret.

Random forest algorithms are a special case of bagging applied to decision trees (CART).
In addition to the Bagging principle, random forests add randomness to variables. For each tree, we select a 
sub-sample by bootstrap of individuals and at each stage, the construction of a node of the tree is done on a 
subset of randomly drawn variables.

The operating principle of random forests is simple: many small classification trees are produced on a random 
fraction of data.
Random Forest then votes these poorly correlated classification trees to infer the order and importance of the 
explanatory variables."""

# ------------------------------------------------------------------------------------------------------------
"""The dataset used in this exercise comes from data from a telecommunications company and contains information on the 
services used and the consumption of 3333 customers.
The 'churn' column identifies customers who terminated their contract with the telecommunications company within 6 months of 
collecting this data.
The term "churn", widely used in marketing, means the loss of customers or subscribers.

The objective of this exercise is to create a model, from a random forest, to predict the possible departure of customers 
from the company within a period of 6 months."""

# ------------------------------------------------------------------------------------------------------------

"""(a) Load package together from sklearn library.
(b) Load the train_test_split package from the sklearn.model_selection library.
(c) Load the pandas library as pd.
(d) Read the churn_dataset.csv file into a DataFrame called churn_df."""


import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split

churn_df = pd.read_csv("data/churn_dataset.csv")

churn_df.head()

churn_df.info()
# ------------------------------------------------------------------------------------------------------------
"""The 'state', 'area code' and 'phone number' columns are personal data which a priori do not influence whether or not to 
unsubscribe from a contract. It is therefore possible, even recommended, to delete them.
The 'international plan' and 'voice mail plan' columns represent subscription to options, and it is better to replace 
these features with indicator variables, because random forests on scikit-learn do not handle qualitative variables."""

"""(f) Separate in target, the churn column from churn_df.
(g) Transform the variable international plan into dummy variables and add them to the columns of churn_df.
(h) Transform the voice mail plan variable into indicator variables and add them to the columns of churn_df.
(i) Create DataFrame data from chun_df after removing 'international plan', 'voice mail plan', 'state', 'area code', 
'phone number' and 'churn' variables.
"""

target = churn_df["Churn?"]

churn_df = churn_df.join(pd.get_dummies(churn_df["Int'l Plan"], prefix="international"))
churn_df = churn_df.join(pd.get_dummies(churn_df["VMail Plan"], prefix="voicemail"))

to_drop = ["Int'l Plan", "VMail Plan", "State", "Area Code", "Phone", "Churn?"]
data = churn_df.drop(to_drop, axis=1)


"""(j) Separate the dataset into training and test sets, so that the test set is 20% of the total data.
(k) Add the argument random_state=12 in the train_test_split function for the reproducibility of the choice of randomness."""

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=12
)
# -------------------------------------------------------------------------------------------------------------------------------
"""The ensemble module allows to build trees in parallel through the n_jobs parameter. This integer specifies the number 
of calculations which will be processed simultaneously and if n_jobs = -1, all the cores available on the machine will 
be used."""

"""(a) Using the [RandomForestClassifier] method of the ensemble module, create a clf Classifier with the parameter: n_jobs=-1. Add the argument random_state=321, for the reproducibility of the choice of randomness.
(b) Train the algorithm on the training set (X_train and y_train)."""

clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321)
clf.fit(X_train, y_train)


"""(c) Predict the test set data and store it in the y_pred variable.
(d) Display the test sample confusion matrix."""

y_pred = clf.predict(X_test)
pd.crosstab(y_test, y_pred, rownames=["Real Class"], colnames=["Predicted Class"])


"""(e) Calculate the rate of correct predictions of the classification using the score function."""

clf.score(X_test, y_test)

# ------------------------------------------------------------------------------------------------------------------------

"""(f) Calculate the probabilities for X_test to belong to each of the two classes. Then store these probabilities in the 
y_probas variable. We will use the predict_proba() method."""

y_probas = clf.predict_proba(X_test)

# ------------------------------------------------------------------------------------------------------------------------
"""The cumulative lift curve, or gain curve, makes it possible to know, thanks to the test sample, the percentage 
of "churners" which will be reached, according to a chosen target size.

The plot_cumulative_gain() function of the metrics sub-module of the scikitplot package allows you to display a 
cumulative gain curve very easily. It suffices to give it as an argument, the vector of the real labels of the test 
sample, and that of the predictions made by the model."""

"""(g) Import matplotlib.pyplot as plt.
(h) Import scikitplot under the diminutive skplt.
(i) Display cumulative gain curve with y_test and y_probas."""

import matplotlib.pyplot as plt
import scikitplot as skplt

skplt.metrics.plot_cumulative_gain(y_test, y_probas, figsize=(12, 8))
plt.show()
