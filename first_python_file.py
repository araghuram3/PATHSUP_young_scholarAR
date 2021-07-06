import numpy as np

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

# Store the feature data
X = data.data# store the target data
y = data.target# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassifier
logreg = KNeighborsClassifier(n_neighbors=6)
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)

print(calcAccuracy(predictions, y_test))