# my first python file
print('Hello World')

# import datasets
from sklearn import datasets
print('Loaded datasets')

# load our breastcancer dataset
bc_dataset = datasets.load_breast_cancer()

# bunch of print statements to visualize the code
# print(bc_dataset.keys())
# # data is input features & target is the labels for supervised learning

# print(bc_dataset.feature_names)
# print(bc_dataset.target_names)
# print(bc_dataset.DESCR)
# print(bc_dataset.filename)

# look a size of dataset
print(len(bc_dataset.data))
print(bc_dataset.data[500])