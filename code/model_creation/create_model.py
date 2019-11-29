# Model Creation (Logistic Regression)

import nltk
from sklearn.linear_model import Logistic Regression


# input: tuple_list with the following format - [([List of parameter values]1, label1), ([List of parameter values]2, label2), etc...]
def make_model(tuple_list):

	# a) creating a LogReg classifier
	classifierLR = nltk.classify.SklearnClassifier(LogisticRegression())
	classifierLR.train(tuple_list)

	return classifierLR

