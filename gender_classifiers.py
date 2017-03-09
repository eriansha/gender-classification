# Compare three algorihms for best result
# import the depedencies
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# our data set
# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
	 [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
	 [159, 55, 37], [171, 75, 42], [181, 85, 43]]


# labels for each data
Y = ['male', 'female', 'female', 'female', 'male', 'male',
	 'male', 'female', 'male', 'female', 'male']