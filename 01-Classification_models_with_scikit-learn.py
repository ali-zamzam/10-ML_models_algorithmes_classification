"""Classification models with scikit-learn"""

# (((((((explanatory variables(valeurs explicatives) --> features))))))))

# ((((((((What are quantitative and qualitative data?))))))))))

# Quantitative data are measures of values or counts and are expressed as numbers.

# Quantitative data are data about numeric variables (e.g. how many; how much; or how often).

# *****************

# Qualitative data are measures of 'types' and may be represented by a name, symbol, or a number code.

# Qualitative data are data about categorical variables (e.g. what type).

# ----------------------------------------------------------------------------------------------------------
"""
- Supervised learning:

- Supervised Learning is an automatic learning technique where one seeks to automatically produce rules that are not defined 
a priori, from a learning database containing examples.

- The main objective of supervised learning is to build and train a statistical model in order to predict output data(s)
 usually called ***label(s)***, based on the input data. usually called ***features***."""
# ----------------------------------------------------------------------------------------------------------

"""
- Types of supervised learning:
There are two main types of supervised learning tasks:

- Regression is the prediction of a quantitative or continuous variable from explanatory variables.
For example: estimate the rent of an apartment from its surface, its location, its exposure, etc.

- The objective of classification is to identify the class(es) to which objects belong based on their descriptive features. 
It is therefore a prediction of a qualitative or categorical variable.
For example: determining whether an email should be classified as clean or spam based on information about the email, such 
as the subject, the body of the text, the presence of hypertext links, etc.

- In a classification problem, labels can be of two types:

- Ordinal when the classes are naturally ordered between them. For example: height, age, test score, etc.
- Cardinal or Nominal, when the classes cannot be classified according to an order. For example: gender, colors, etc..""".
# ----------------------------------------------------------------------------------------------------------
"""
Learning mechanism
In supervised learning, data must be systematically represented in the form of a feature matrix, which stores the 
attributes or criteria used to train a model, and a label vector, which contains the classes corresponding to each sample.


In order to have an intuition on the quality and performance of a classification model, 
you must first separate the dataset 
into two:

- The first set is the training set, which is the learning base on which the model is trained.
- The second set is the test set, on which the performance of the model is calculated. In both cases, the classes corresponding 
to each sample are known a priori."""
# ----------------------------------------------------------------------------------------------------------

"""After launching the learning mechanism, it is then possible to make predictions for the data present in the test set, 
and build a confusion matrix, which distinguishes all cases of good or bad classification for each class. Most of a model's 
performance scores and metrics are calculated from this matrix.
"""
"""The confusion matrix allows to evaluate the classification model and thus to select the best models and parameters to use."""
#                           Positive                    Negative

# Positive                Vrai Positive               faux Negative

# Negative                Faux Positive               Vrai Negative

# ----------------------------------------------------------------------------------------------------------
"""
- Classification into multiple classes

- Although most machine learning techniques concern cases of binary classification, it is possible to address "multi-class" 
problems.
In general, a multi-class problem is decomposed into several binary subproblems according to one of the two possible 
strategies presented below:

- One against all (One-vs-all):
Individuals of a given class are separated from all others. In other words, elements of one class are compared with 
other elements of other classes. The class of an individual is then given from the results of each binary classification 
sub-model.

- One on one (One-vs-one):
In this method (OvO), for a classification problem with KK classes, we train K(K-1)/2 binary classifiers; 
each receives the samples of a pair of classes from the original training set, and must learn to distinguish between 
these two classes. At prediction time, a voting scheme is applied: all K(K-1)/2 classifiers are applied to a new 
sample and the class that obtained the highest number of predictions is chosen by the combined classification model.
"""
# ----------------------------------------------------------------------------------------------------------
"""In many cases, it is appropriate to discretize quantitative variables to make machine learning algorithms 
more efficient.

**For example, we can discretize the ages of people into several categories 
(15-18 years old, 18-25 years old, 25-40 years old, etc). This technique is widely used by statisticians to:**

- Harmonize the type of variables.
-Correct very skewed distributions.
- Reduce the role of extreme values.
- Use statistical techniques that only work with qualitative variables.

(The ***pandas cut function*** allows you to cut a quantitative variable into classes. It thus makes it possible to 
discretize continuous variables according to bounds defined by the user.)

***The arguments of the function are:***

- x: A Series (column of DataFrame) to discretize.
- bins: A list containing the bounds that we wish to use for the discretization.
- labels: A list of character strings that will serve as a label to give to the categories built.

*****The number of labels must therefore be equal to (number of terminals -1).*****
***This function returns a Series containing the discretized column.***


The choice of limits for the division into classes can be done by logic.
For instance :

If a person's age is less than 18, then the individual is placed in the class of minors.
Conversely, if the age is greater than 18, then the individual is placed in the class of adults.
Alternatively, we can use statistical arguments to create our categories.
For instance :

- Create classes of the same size.
- Create classes with equal amplitudes.
- Create classes with the method of nested averages: A first average divides the individuals in two then each group 
is again divided in 2 by its respective average and so on."""

# ------------------------------------------------------------------------------------------------

"""Once the quantitative variables have been discretized, it is common to proceed to their dichotomization.

Suppose we have the Gender column describing the gender of an individual.

Gender
Women
Man
Man
Women
To dichotomize a variable amounts to transforming each modality of the variable into a new indicator variable, 
indicating whether or not the modality in question corresponds to the individual.
Thus, the dichotomization of the Genre column gives two new columns Genre_Homme and Genre_Femme:

Gender Gender_Female    Gender_Male
Woman   1                   0
Man     0                   1
Man     0                   1
Woman   1                   0


- Dichotomization is necessary for machine learning models of linear types that we will see in the following. Indeed, 
these models are incapable of interpreting qualitative variables. Thanks to the dichotomization we have transformed 
this qualitative variable into a "quantitative" variable that can be interpreted by a machine learning model.

- NB: You will more often find the term One Hot Encoding, to talk about dichotomization. Moreover, One Hot Encoding 
is not the only technique to encode a qualitative variable.

- Pandas' get_dummies function makes it easy to transform a qualitative variable into as many indicator variables 
as it contains categories.

It takes as argument:

- data: a Series or a DataFrame to dichotomize.
- prefix: The prefix to add to the names of created variables. By default, the prefix will be the name of the column 
to dichotomize."""

"""((((Note: When a DataFrame contains non-numeric categorical variables, applying the get_dummies() function to 
the entire DataFrame will remove these variables and replace them with the corresponding indicator variables.)))"""
# --------------------------------------------------------------------------------------------------------------

"""In order to test the performance of the classification model, it is necessary to select a part of the data which is 
dedicated to the evaluation and which, consequently, is not taken into account in the training of the model.

To do this, the data must be systematically divided into a training set (X_train and y_train) and a test set 
(X_test and y_test).

- train_test_split is a very useful function of the model_selection submodule of Scikit-learn for splitting data. 
- The function separates the feature and label matrices passed as parameters into a training set and a test set.
- The test_size parameter allows you to choose in which proportions the data is distributed.
- The shuffle parameter allows you to decide whether the samples are chosen randomly (default option), or whether 
the separation must be done respecting the order of the data indicated. This last case can correspond to use cases
 where the processed data admits an order or a temporal dimension such as a financial course, a user's journey on 
 a site, etc...

- Usually, the test set size is between 15% and 30% of the total amount of data. The choice of distribution depends 
 essentially on the quantity and quality of the data available."""
# ---------------------------------------------------------------------------------------------------------------
"""Accuracy is not the only metric for evaluating model performance.
The confusion matrix also allows us to evaluate the precision and the recall or the fβfβ -score which is a 
harmonic mean of the precision and the recall.

Positive class recall is also called "sensitivity" (or true positive rate) and negative class recall 
"specificity" (or true negative rate)

These two measures should be carefully evaluated. Indeed, the sensitivity and the specificity must be examined jointly, 
a high value of one of them alone cannot be a sign of the good performance of a model.
A model that, for example, classifies all items as positive will indeed have a sensitivity of 1 but a zero specificity, 
and vice versa.

The classification_report() function of the sklearn.metrics sub-module allows you to display some of these additional 
metrics, with as arguments the vector of the true labels, and that of the predicted labels."""
# ----------------------------------------------------------------------------------------------------------------------------------
"""ROC,AUC"""

"""Another very effective tool for evaluating the performance of a model is the ROC curve. 
The ROC curve (for Receiver Operating Characteristic) is the ideal tool to summarize the performance of a 
binary classifier according to all possible thresholds. It avoids a long work of class predictions for different 
thresholds, and evaluation of the confusion matrix for each of these thresholds.

Graphically, the ROC measurement is represented in the form of a curve which gives the rate of true positives, 
the sensitivity, according to the rate of false positives, the antispecificity (= 1 - specificity). 
Each classification threshold value will provide a point on the ROC curve, which will go from (0, 0) to (1, 1).

The closer the curve gets to the (0,1) point (top left), the better the predictions. A model with sensitivity 
and specificity equal to 1 is considered perfect.

The area under the curve (AUC: Area Under the Curve) is very useful. In a single number it summarizes the ability 
of the model to distinguish the negative class from the positive class (Not Admitted / Admitted).

An AUC score of 0.5 means that the model is no better than a random classification, an AUC score of 1.0 means 
a perfectly predictive model, and an AUC of 0.0 is perfectly anti-predictive (very rare).

The sklearn.metrics module contains the roc_curve() function which returns a table containing the false positive 
rates (antispecificity), a table of the true positive rates (sensitivity), and a table of classification thresholds 
taking values between 0 and 1. It takes as argument the vector of labels that we want to predict, a vector of the 
probabilities of belonging to the positive class and the argument pos_label, which allows to choose which label is 
defined as positive.

The auc() function of the same module calculates the area under the curve when given as arguments a vector of false 
positive rate and a vector of the same size of true positive rate."""
# ------------------------------------------------------------------------------------------------
"""Feature Scaling(Normalization)"""

"""
- Standardization, or data normalization, is the process of subtracting the mean for each variable and then 
dividing it by the standard deviation.

- Center-reduce variables , transform them into units compatible with a distribution of mean 0 and 
standard deviation 1, independent of their original distributions and units of measurement.

- Standardization/Normalization is a common task in Machine Learning. Many algorithms assume that features are 
centered around 0 and have approximately equal variance."""

# example

# scaler = preprocessing.StandardScaler().fit(X_train)

# X_train_scaled = scaler.transform(X_train)

# print(X_train_scaled.mean(axis=0))
 
# print(X_train_scaled.std(axis=0))

# Numpy array descriptive statistics methods apply to columns only if they receive axis=0 as argument.

"""Algorithms need scaling"""
# Linear/Non Linear Regression
# Logistic Regression
# KNN
# SVM
# Neural Networks
# K-means clutering
# PCA
# SVD
# Factorization Machines

"""No need"""
# Random forest
# Gradient Boosted decision trees
# Naîve Bayes
# -----------------------------------------------------------------------------------------------
"""hyperparameters"""

"""Many machine learning algorithms, such as SVMs, rely on hyperparameters that are not always easy to determine 
to obtain the best performance on a data set to be processed.

Unlike the simple parameters of the model which derive directly from the data (for example: the coefficients of a 
regression) the hyperparameters make it possible to decide on the structure of the model and are to be adjusted 
before training it.

- In the majority of cases, when several hyperparameters need to be tuned, and you don't know which ones to use to get 
the best possible model, the most effective strategy is to create a search grid.
- The parameters to be varied are indicated. Then, thanks to the GridSearchCV() function of the model_selection module, 
the parameters are crossed and a model is created then evaluated for each possible combination by cross-validation."""

# Example (svm):

# parametres = {'C':[0.1,1,10], 'kernel':['rbf','linear', 'poly'], 'gamma':[0.001, 0.1, 0.5]}

# grid_clf = model_selection.GridSearchCV(estimator=clf, param_grid=parametres)

# grille = grid_clf.fit(X_train_scaled,y_train)

# print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']]) 
# print(grid_clf.best_params_)
# ---------------------------------------------------------------------------------------------------------
"""Types of Distance Metrics in Machine Learning
Euclidean Distance
Manhattan Distance
Minkowski Distance
Hamming Distance
chebyshev Distance"""

# ---------------------------------------------------------------------------------------------------------
"""use csv from uci"""

# df = pd.read_csv("url of data set/ the name of file to be used", names = [Attribute Information(in the dataset page)])
# (shoud be names not name)

# df = pd.read_csv(
#     "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
#     names=[
#         "radius",
#         "textures",
#         "perimeter",
#         "area",
#         "smoothness",
#         "compactness",
#         "concavity",
#         "concave points",
#         "symmetry",
#         "fractal dimension",
#     ],
# )

# df.head()
# ---------------------------------------------------------------------------------------------------------
"""in Desicion tree:
To find out which variables most determined the 'diagnosis' of each cell mass, the feature_importances_ attribute 
returns the normalized importance of each variable in building the tree.
In scikit-learn importance is defined as the total decrease between one node and the next two of the impurity 
criterion used to split the node. The greater the difference between the calculated impurity for a node and its 
'child' nodes, the greater the variable used to divide the node.
"""

# example:
# feats = {}
# for feature, importance in zip(data.columns, dt_clf.feature_importances_):
#     feats[feature] = importance 
    
# importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
# importances.sort_values(by='Importance', ascending=False).head(8)
# ---------------------------------------------------------------------------------------------------------
"""boosting and bagging"""

"""are both ensemble methods that produce N number of estimators to get one, but while they are independent for Bagging, 
Boosting creates models that iteratively improve by emphasizing where previous models have failed.
generate different datasets by resampling, but whereas resampling is totally random for Bagging, Boosting calculates 
different weights to select at each stage the most difficult observations to predict.
both determine the final decision by carrying out a majority vote or an average on the N estimators, but the average is 
equally weighted for Bagging, and weighted by coefficients relating to the performance of the estimators for Boosting.
are good at reducing variance and providing greater stability, but only Boosting attempts to reduce bias, while Bagging 
is better at avoiding the overfitting that Boosting can sometimes create."""


"""
- Boosting is a set of methods essentially aimed at reducing the bias of simple and weak Machine Learning models and 
converting them into a stable and powerful model.

The general principle of boosting consists in building a family of "weak" estimators built recursively, which are then 
aggregated by a weighted average of the estimates (in regression) or a majority vote (in classification).
By weak, it is understood a decision rule whose error rate is slightly better than that of a purely random rule.

Each estimator is an improved version of the previous one, which aims to give more weight to ill-fitting or ill-predicted 
observations. Thus at each iteration, the evaluation of the estimator allows resampling of the data, with greater weight 
given to poorly predicted observations. The estimator built at stage i will therefore concentrate its efforts on the 
observations that are ill-adjusted by the estimator at stage i − 1.
Finally the classifiers are combined, and weighted by coefficients associated with their respective predictive performances.

There are many Boosting algorithms.
The most popular is the AdaBoost (for Adaptive Boosting) algorithm developed by Freund & Schapire (1997).
Its operation is as follows:

A "weak" classification rule is chosen. The idea is to apply this rule several times by judiciously assigning a different 
weight to the observations at each iteration.
The weights of each observation are initialized to 1n1n ( nn being the number of observations) for the estimation of the 
first model.
They are then updated for each iteration. The importance of an observation is unchanged if the observation is well classified; 
otherwise it increases with the measured goodness-of-fit of the model.
The final aggregation is a combination of the estimators obtained weighted by the goodness of fit of each model.
The sklearn.ensemble package makes it possible to implement the AdaBoost algorithm in the case of multi-class classification, 
in particular thanks to the AdaBoostClassifier class which makes it possible to create a classifier using by default a simple 
decision tree as initial classification rule."""


"""
- Bagging or Bootstrap AGGregatING
The term Bagging comes from the contraction of Bootsrap Aggregating, it brings together a set of methods introduced by 
Léo Breiman (1996) aimed at reducing the variance and increasing the stability of Machine Learning algorithms used for 
classification or regression.

The general Bagging method consists mainly of training a model on different subsets of the same size as the initial sample, 
using the Bootstrap technique, ie random draw with replacement.
The method therefore builds a set of independent estimators, unlike Boosting, which are then aggregated (or bagged) into 
a meta-model, with a majority vote for the classification, and an average for the regression.

Unlike Boosting, choosing a large number of estimators will not lead to an additional risk of overfitting.
Indeed, the higher the number of estimators, the more the bias of the final model will be equivalent to the average of the 
aggregated biases and the variance will decrease all the more as the estimators that are aggregated are decorrelated. It will 
therefore be in our interest to choose the highest possible number of estimators, depending on the time we wish to grant to 
the training process.

 For bagging to have any meaning, the chosen estimators must therefore be vulnerable to variations in the original sample by 
 bootstrap. Regression and classification trees are known to be particularly unstable and are therefore ideal estimators for 
 bagging. Most algorithms also use default trees in their bagging procedures.
The prediction error calculated, in general, for bagging methods is the so-called Out Of Bag (OOB) error, i.e. for each 
observation, the average of the errors is calculated for all the models trained on a bootstrapped sample of which she is not 
a part. This technique helps prevent over-fitting.

The BaggingClassifier class of the sklearn.ensemble package allows to create a classifier using the Bagging algorithm from 
default classification trees.

Bagging is a simple and robust ensemble method that reduces variance when predictors are unstable. Its prediction error 
estimation by Bootstrap prevents over-fitting.

The use of Bagging is suitable for high-variance algorithms which are thus stabilized, in particular neural networks and 
decision trees.
However, it can also degrade qualities for more stable algorithms, e.g. kk nearest neighbors method, or linear regression"""
# ------------------------------------------------------------------------------------------------
"""Voting Classifier"""

"""The Voting Classifier is a scikit-learn meta-classifier allowing to combine several Machine Learning estimators, 
similar or conceptually different.
Precisely, it is a question of constituting a college of experts which is represented by models such as decision trees, 
the method of the kk nearest neighbors or logistic regression, then to make them vote.
The VotingClassifier class of scikit-learn allows to perform a 'hard' or 'soft' vote.

In the 'hard' vote, each classification model predicts a label, and the final label produced is the one predicted most 
frequently.
In 'soft' voting, each model returns a probability for each class, and the average of the probabilities is calculated to 
predict the final class (only recommended if the classifiers are well calibrated).
In both cases, it is possible to assign a weight to each estimator, allowing more importance to be given to one or more 
models."""
# ------------------------------------------------------------------------------------------------
"""stacking"""

"""Stacking is an ensemble method whose principle is the simultaneous training of various Machine Learning algorithms, the 
results of which are then used to train a new model which makes it possible to optimally combine the predictions of the first 
estimators.
This method is based on the following technique:

The first step consists of specifying a list of L base algorithms and corresponding parameters, as well as the meta-learning 
algorithm.
Each of the L algorithms is then trained on the training set, containing N observations.
A cross-validation procedure makes it possible to obtain the predictions of each of the models for the N observations.
The meta-learning algorithm is then trained on this collected data and makes new predictions.

The ensemble model ultimately consists of the set of L base algorithms and the meta-learning model and can be used to 
generate predictions on a new dataset."""
# ------------------------------------------------------------------------------------------------
"""XGBoost (EXtreme Gradient Boosting) is one of the most popular Machine Learning libraries in recent years.
It is a staple of Kaggle-like competitions, which is suitable for both classification and regression.
It is based on the Gradient Boosting technique, and can directly implement parallel computing on any machine, which 
makes it much more efficient than other packages using similar methods.

Gradient Boosting is a generalization of Boosting in which the loss function is used similarly to gradient descent.

 As a reminder, gradient descent minimizes a loss function by following the “slope” of this function using its 
 gradient or an approximation of it.
By design, this boosting technique underlyingly uses decision trees.
The main idea is to aggregate several models created iteratively, but also to give a different weight to each of them.

The following approach explains the reasoning used in the design of a GBT:

Take a random weight (wiwi weight) for the weak classifiers (aiai parameters) and train a final classifier.
Compute the error induced by this final classifier, and find the weak classifier that comes closest to this error.
Subtract this weak classifier from the final classifier while optimizing its weight with respect to a loss function.
Repeat the process iteratively.
The gradient boosting procedure therefore consists in finding the weights which optimize the cost function relating 
to the classification problem.
It is therefore a question of exploring a space of simple functions by a gradient descent.

XGBoost uses a variation of gradient descent called functional gradient descent, since we are working in a functional 
space when talking about classifiers in the boosting algorithm.
The gradient of the loss function is used to calculate the weights of individuals during the construction of each new model.

XGBoost offers to save the matrices or the models built and to reload them later, to avoid restarting the same expensive 
calculations several times."""

# --------------------------------------------------------------------------------------------------------------------
"""The confusion matrix: each column of the matrix represents the number of occurrences of an estimated class, while 
each row represents the number of occurrences of an actual class.

The recall is defined by the number of relevant documents found with regard to the number of relevant documents that 
the database has.

Accuracy* is the number of relevant documents found compared to the total number of documents proposed by the search 
engine for a given query.

The F-measure is a popular metric that combines precision and recall by calculating their harmonic mean."""