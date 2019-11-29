# Model Creation (Logistic Regression)

import nltk
from sklearn.linear_model import Logistic Regression

def make_model(tuple_list):

	# a) creating a LogReg classifier
	classifierLR = nltk.classify.SklearnClassifier(LogisticRegression())
	classifierLR.train(tuple_list)

	return classifierLR

def 