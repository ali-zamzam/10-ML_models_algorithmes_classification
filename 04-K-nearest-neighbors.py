"""KNN"""

"""The K-nearest neighbor (KNN) method is a supervised learning method that is extremely simple to implement 
in its most basic form, and yet often performs well for complex classification tasks. The KNN algorithm does 
not train on any data, but each time uses all the data it has to classify new data.

The principle is as follows: a datum of unknown class is compared to all the stored data. The class to which 
the new datum is attributed is the majority class among its K nearest neighbors in the sense of a chosen 
distance. By default, the distance used by the KNeighborsClassifier class is the Euclidean distance."""

# example :
# ( [] [] [] ** [] [] () ) the ** take the attribute of [] [] and now --> ( [] [] [] [] [] [] ) 
# because [] is the nearest of ** (not the () ) and ther's []  more than () 
# --------------------------------------------------------------------------------------
"""The data used in this exercise comes from a collection of figures written by hand by a group of 43 people, available here

(https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits)

and directly importable into the 'datasets' package of scikit-learn.
These are black and white images, normalized, centered and of size 8x8 pixels. Images are given here as 
one-dimensional 
vectors of size 64 (pixels) in the data attribute, and as arrays of size 8x8 in the images attribute. 
The target attribute 
contains the labels corresponding to each image (a number between 0 and 9)."""

# -----------------------------------------------------------------------------------------------
"""Data preparation and modeling"""

"""
(a) Load the neighbors package from the sklearn library.
(b) Load the datasets package from the sklearn library.
(c) Load the train_test_split package from the sklearn.model_selection library.
(d) Load the pandas library as pd, and numpy as np.
(e) Import the data dictionary into digits using the datasets.load_digits() function.
(f) Save in a DataFrame X_digits the data contained in the data attribute of digits.
(g) Save in y_digits the labels contained in the target attribute of digits."""



import numpy as np
import pandas as pd
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()

X_digits = pd.DataFrame(digits.data)
y_digits = digits.target



"""(h) Execute the following cell to randomly display six pixelated digits present in the data."""

%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm  # to import new color maps

j=0

for i in np.random.choice(np.arange(0, len(y_digits)), size=6):
    j=j+1
# We store the index in the list i to be able to display the corresponding label later.
    
    plt.subplot(2,3,j)
# Adding *plt.subplot(2,3,j)* at each iteration displays all images
# sets on the same figure.

    plt.axis('off')
# Allows you to remove the axes (here is used to better see the titles)
    
    plt.imshow(digits.images[i],cmap = cm.binary, interpolation='None')
# # Displays image n°i
# Using cm.binary shows numbers in gray on a white background.

    plt.title('Label: %i' %y_digits[i])
# For each image we write in title the label which corresponds to it. 

# ----------------------------------------------------------------------------------------------------
"""(i) Divide the matrices X_digits and y_digits into an 80% training set (X_train, y_train) and a 20% test set 
(X_test, y_test).
Add the argument random_state = 126 in the train_test_split function for the reproducibility of the choice 
of randomness."""

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=126)
# ----------------------------------------------------------------------------------------------------
"""Types of Distance Metrics in Machine Learning
Euclidean Distance
Manhattan Distance
Minkowski Distance
Hamming Distance"""

"""
(b) Create a knn classifier, with k = 7 and the 'minkowski' distance 

(default distance, with p=2, Euclidean distance).

(c) Fit the classifier on the training set (X_train and y_train)."""

knn = neighbors.KNeighborsClassifier(n_neighbors=7, metric='minkowski')
knn.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------------
"""Evaluation of the classification model"""

"""(a) Apply the model to the test set data and store the obtained predictions in the y_pred variable.
(b) Display a confusion matrix to compare actual and predicted classes.
"""
y_pred = knn.predict(X_test)
pd.crosstab(y_test, y_pred, rownames=['Real Class'], colnames=['predicted Class'])

# (The created model already seems to perform well enough on the test set.
# Out of 360 written digits tested, only three 5 digits and four 9 digits were misclassified.)


"""by choosing a different distance, or another number of 'neighbors'.

(c) Create a new classifier knn_m, with k=5 and distance 'manhattan'.
(d) Fit the new classifier on the training set (X_train and y_train)."""

knn_m = neighbors.KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_m.fit(X_train, y_train)

y_pred = knn_m.predict(X_test)
pd.crosstab(y_test, y_pred, rownames=['Real Class'], colnames=['predicted Class'])

"""(e) Calculate the performance score ('accuracy') for the two models."""

score_minkowski = knn.score(X_test, y_test)

score_manhattan = knn_m.score(X_test, y_test)

print(score_minkowski, score_manhattan)

# ------------------------------------------------------------------------------------------------------
"""(f) Create three lists score_minko, score_man, score_cheb, in which you will store the scores of 3 models 
using the Minkowski, Manhattan and Chebyshev matrices respectively, for values of k ranging from 1 to 40."""

score_minko = []
score_man = []
score_cheb = []

for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score_minko.append(knn.score(X_test, y_test))

for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)
    score_man.append(knn.score(X_test, y_test))
    
for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
    knn.fit(X_train, y_train)
    score_cheb.append(knn.score(X_test, y_test))


"""(g) Display in a graph the lists created as a function of the value of k.
(h) Use different colors and legends to differentiate metrics."""

import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(range(1, 41), score_minko, color='blue', linestyle='dashed', lw=2, label='Minkowski')
plt.plot(range(1, 41), score_man, color='orange', linestyle='dashed', lw=2, label='Manhattan')
plt.plot(range(1, 41), score_cheb, color='red', linestyle='dashed', lw=2, label='Chebyshev')
plt.title('Score - valeur de K')  
plt.xlabel('Valeur de K')  
plt.ylabel('Accuracy') 
plt.legend();

"""Minkowski and Manhattan distances give better performance when k is small (<10). The Minkowski distance seems more 
stable, and its rate of good predictions only decreases from k=20."""
