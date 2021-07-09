import numpy as np
import os
# from statistics import NormalDist
url='https://raw.githubusercontent.com/python/cpython/3.8/Lib/statistics.py'

import urllib.request 
urllib.request.urlretrieve(url, os.path.basename(url)) 
from statistics import *

def calcAccuracy(predictions, labels):
	# two methods to evaluate accuracy

	# method 1
	# return np.sum(predictions == labels)/float(len(predictions))
	
	# method 2
	counter = 0
	for ind in range(len(predictions)):
		pred = predictions[ind]
		label = labels[ind]
		if pred == label:
			counter += 1

	return float(counter)/float(len(predictions))


# my first python file
print('Hello World')

# import datasets
from sklearn import datasets
print('Loaded datasets')

# load our breastcancer dataset
data = datasets.load_breast_cancer()
print(data.feature_names)

# Store the feature data
X = data.data # store the target data
y = data.target # split the data using Scikit-Learn's train_test_split

inds_0 = y == 0
x_0 = X[inds_0, :]
x_1 = X[~inds_0, :]

mean_x0 = np.mean(x_0, axis=0)
mean_x1 = np.mean(x_1, axis=0)
var_x0 = np.var(x_0, axis=0)
var_x1 = np.var(x_1, axis=0)
for ind in range(len(mean_x0)):
	overlap = NormalDist(mu=mean_x0[ind], sigma=np.sqrt(var_x0[ind])).overlap(NormalDist(mu=mean_x1[ind], sigma=np.sqrt(var_x1[ind])))
	print("Param", str(ind),  data.feature_names[ind], ":", str(overlap))

# print("x_0 size: ", x_0.shape)
# print("x_1 size: ", x_1.shape)

# param_inds = [27, 20, 22, 7, 2, 3, 23, 0, 12, 13] # good params
param_inds = [14, 11, 8, 21, 18, 17, 11, 9, 4, 28] # bad params
print("Params to use: ", str([ind for ind in param_inds]))
X = X[:, param_inds]

print("Shape of X originally is: ", X.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassifier
logreg = KNeighborsClassifier(n_neighbors=6)
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)

print(calcAccuracy(predictions, y_test))