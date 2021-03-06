import numpy as np
import matplotlib.pyplot as plt

# import os
# url='https://raw.githubusercontent.com/python/cpython/3.8/Lib/statistics.py'
# import urllib.request 
# urllib.request.urlretrieve(url, os.path.basename(url)) 
# from statistics import *

from statistics import NormalDist

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

################ new stuff ##########################
inds_0 = y == 0
x_0 = X[inds_0, :]
x_1 = X[~inds_0, :]

mean_x0 = np.mean(x_0, axis=0)
mean_x1 = np.mean(x_1, axis=0)
var_x0 = np.var(x_0, axis=0)
var_x1 = np.var(x_1, axis=0)
overlap_vec = []
for ind in range(len(mean_x0)):
	overlap = NormalDist(mu=mean_x0[ind], sigma=np.sqrt(var_x0[ind])).overlap(NormalDist(mu=mean_x1[ind], sigma=np.sqrt(var_x1[ind])))
	overlap_vec.append(overlap)
	print("Param", str(ind),  data.feature_names[ind], ":", str(overlap))

# print("x_0 size: ", x_0.shape)
# print("x_1 size: ", x_1.shape)

plt.plot(range(len(overlap_vec)),overlap_vec)
plt.xlabel('Parameter',fontname='times new roman')
plt.ylabel('Overlap',fontname='times new roman')
plt.yticks(fontname = "Times New Roman") 
plt.xticks(fontname = "Times New Roman") 
plt.show()

param_inds = [27, 20, 22, 7, 2, 3, 23, 0, 12, 13] # good params
# param_inds = [14, 11, 8, 21, 18, 17, 11, 9, 4, 28] # bad params
print("Params to use: ", str([ind for ind in param_inds]))
X = X[:, param_inds]

print("Shape of X originally is: ", X.shape)
####################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassifier
knn_param_vec = []
params = range(2,20)
for p in params:

	logreg = KNeighborsClassifier(n_neighbors=p)
	logreg.fit(X_train, y_train)
	predictions = logreg.predict(X_test)
	knn_param_vec.append(calcAccuracy(predictions, y_test))

fontsize = 14
fontname = 'times new roman'

plt.plot(params,knn_param_vec)
plt.xlabel('Number of neighbors',fontname=fontname,fontsize=fontsize)
plt.ylabel('Accuracy',fontname=fontname,fontsize=fontsize)
plt.yticks(fontname=fontname,fontsize=fontsize) 
plt.xticks(fontname=fontname,fontsize=fontsize) 
plt.show()


plt.bar(range(3),np.array(knn_param_vec[0:3]),tick_label=["0","1","2"])
plt.xlabel('Number of neighbors',fontname=fontname,fontsize=fontsize)
plt.ylabel('Accuracy',fontname=fontname,fontsize=fontsize)
plt.yticks(fontname=fontname,fontsize=fontsize) 
plt.xticks(fontname=fontname,fontsize=fontsize) 
plt.show()