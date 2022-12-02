"""A major advantage of decision trees is that they can be computed automatically from databases by supervised learning 
algorithms.
These algorithms automatically select discriminating variables from unstructured and potentially large data. They can 
thus make it possible to extract logical rules of cause and effect (determinisms) which did not initially appear in the 
raw data."""

"""((((In Machine Learning, a decision tree can be described as a visual representation of an algorithm for classifying 
data according to different criteria called decisions (or nodes).
Each node corresponds to a test on a learning variable, and each of the following branches represents a result of this 
test. Each leaf of the tree (terminal nodes) contains a value of the target variable (a label in the case of a 
classification).

During model training, nodes are created from "optimal" tests against the training set, and the training set ends when 
the leaves of the tree are homogeneous or satisfy a certain criterion of stop.

This decision tree therefore makes it possible, after training on a set of data, to easily make predictions in the form 
of successive logical classification rules. The results are thus easily interpretable and therefore usable, communication 
around the modeling easier. It is therefore a highly appreciated classifier used in business.

The construction of a decision tree is done in principle in 2 phases:

The first phase consists of the construction of the nodes:

From a training set, a recursive process of dividing the data space into increasingly pure sub-samples in terms of 
classes is triggered, on the basis of a predefined criterion.
The classification problem is thus broken down into a series of (nested) tests relating to a variable, of the 
"X>=threshold" type.
On each node, the best test is selected according to a certain criterion (often based on information theory, and in 
particular on the notion of entropy), the objective of which is to reduce the mixing of classes as much as possible 
within each subset created by the different test alternatives.
This results in a succession of classification rules in the form of a tree, each extremity (or "leaf") of which indicates 
the membership of a class.
The class allocated to a leaf is determined by the class most represented among the data of the training set which 
"falls" in this leaf.
The objective of this phase is to generate a hierarchical sequence of tests, as short as possible, which successively 
divides the set of training data into disjoint subsets, such as subgroups of cases belonging to the same class. 
are quickly detected.

The second phase is the pruning phase:

It consists of removing unrepresentative branches to maintain good predictive performance. This step requires the 
creation of a criterion/metric to designate the branches to be pruned, which will depend on the algorithm used.
After pruning, the branches are replaced by terminal nodes, labeled on the basis of the distribution of the training 
data (majority class).
In general, pruning is done from the bottom to the top of the tree ("bottom-up"). It is based on an estimate 
(cross-validation, new sample, statistical estimate, etc.) of the classification error rate: a tree is pruned at 
a certain node if the estimated error rate at this node (by allocating the majority class) is lower than the error 
rate obtained by considering the terminal subtrees.
The pruning continues successively (starting from the extremities) until all the remaining sub-trees satisfy the 
condition on the classification error rates.))))"""

"""In this exercise we will study an example of using decision trees in Python.
Packages used will be pandas, scikit-learn and its sub-packages including tree and model_selection."""


"""The dataset used in this exercise comes from the UCI ML Repositry) and contains data calculated from digitized images
 of breast masses.
They describe the characteristics of the cell nuclei present in each image (radius, perimeter, texture etc.). 
The first column gives the result of the diagnosis of each cell mass: 'B' for benign, 'M' for malignant.
The objective of the exercise is to build a model in the form of a decision tree, to predict whether a cell mass 
is benign or malignant, depending on the characteristics calculated from the image of its biopsy."""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

bc_data = pd.read_csv("data/Breast_Cancer_Wisconsin.csv", index_col=0)

bc_data = bc_data.iloc[:, :-1]

bc_data.head()


bc_data.info()

"""Create the Data Frame data containing the different features and the target vector containing the target 
variable 'diagnosis'."""

data = bc_data.drop("diagnosis", axis=1)

target = bc_data.diagnosis

# -----------------------------------------------------------------------------------------------
"""Data learning"""

"""criterion: This parameter determines how the impurity of a split will be measured. 
The default value is “gini” but you can also use “entropy” as a metric for impurity. splitter: 
This is how the decision tree searches the features for a split. The default value is set to “best”.

- In summary, the entropy is 0 if all the samples of a node belong to the same class, and the entropy is 
maximum if we have a uniform class distribution (i.e. when all the classes of the node have an equal 
probability).

The Gini index is similar to entropy, but the choice of criterion used sometimes gives different 
classifications.
"""

"""(a) Separate the dataset into training and test sets, so that the test set is 20% of the total data. 
Add the argument random_state=123 in the train_test_split function for the reproducibility of the choice of randomness.

(b) Create a DecisionTreeClassifier instance called dt_clf, with criteria criterion='entropy' and argument 
max_depth=4 to specify the maximum number of separation points possible before reaching a "leaf" node. Again, 
add the random_state=123 argument for reproducibility of results.

(c) Train the classifier on the training set."""

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=123
)

dt_clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=123)
dt_clf.fit(X_train, y_train)

# -----------------------------------------------------------------------------------------------
"""(e) Apply the model to the test set data and store the obtained predictions in the y_pred variable.
(f) Display a confusion matrix to compare actual and predicted classes."""

y_pred = dt_clf.predict(X_test)

pd.crosstab(y_test, y_pred, rownames=["real class"], colnames=["predicted class"])

# ---------------------------------------------------------------------------------------------------------

"""(g) Display the 8 most important variables for dt_clf, along with their respective importances."""

feats = {}
for feature, importance in zip(data.columns, dt_clf.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient="index").rename(
    columns={0: "Importance"}
)
importances.sort_values(by="Importance", ascending=False).head(8)
# ---------------------------------------------------------------------------------------------------------
"""(h) Create a dt_clf_gini classifier, having as parameters: criterion='gini' , max_depth=4 and 
random_state=321.
(i) Train the new model on the training set (X_train and y_train).
(j) Save model predictions on X_test in y_pred.
(k) Display the corresponding confusion matrix."""

dt_clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=321)
dt_clf_gini.fit(X_train, y_train)
y_pred = dt_clf_gini.predict(X_test)
pd.crosstab(y_test, y_pred, rownames=["real class"], colnames=["predicted class"])


feats = {}
for feature, importance in zip(data.columns, dt_clf_gini.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient="index").rename(
    columns={0: "Gini-importance"}
)

# Affichage des 8 variables les plus importantes
importances.sort_values(by="Gini-importance", ascending=False).head(8)
# ---------------------------------------------------------------------------------------------------------

from sklearn import tree

tree.plot_tree(dt_clf_gini)


import matplotlib.pylab as plt

neighbors_setting = range(1,15)
training_accuracy = []
test_accuracy = []

max_dep = range(1,15)

for md in max_dep:
    tree = DecisionTreeClassifier(max_depth=md,random_state=0)
    tree.fit(X_train,y_train)
    training_accuracy.append(tree.score(X_train, y_train))
    test_accuracy.append(tree.score(X_test, y_test))
 
plt.plot(max_dep,training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_setting,test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Max Depth')
plt.legend()
