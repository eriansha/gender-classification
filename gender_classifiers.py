# Compare three algorihms for best result
# import the depedencies
from sklearn.neural_network import MLPClassifier
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

# training data set into 3 algoritms
# Decision Tree
train_nn = MLPClassifier(solver= 'lbfgs', alpha = 1e-5, hidden_layer_sizes=(5,2), random_state=1)
train_tree = train_clf.fit(X, Y)

# Support Vector Machine
train_svm = svm.SVC()
train_svm = train_svm.fit(X, Y)

# Gaussian Naive Bayes
train_gnb = GaussianNB()
train_gnb = train_gnb.fit(X, Y)

# testing model with a new list bodymetrix
_X = [[169, 61, 42], [155, 45, 38], [165, 65, 41], 
	[170, 75, 43],[183,83,44],[166,47,36]]

_Y = ['male', 'female', 'male', 'male', 'female', 'female']

# predict model with Neural Network
predict_nn = train_nn.predict(_X)

# predict model with SVM
predict_svm = train_svm.predict(_X)

# predict model with Gaussian Naive Bayes
predict_gnb = train_gnb.predict(_X)
