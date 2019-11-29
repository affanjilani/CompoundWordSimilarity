# Model Creation (Logistic Regression)

import nltk
from sklearn.linear_model import Logistic Regression
from sklearn.model_selection import GridSearchCV
import numpy as np


# input: tuple_list with the following format - [([List of parameter values]1, label1), ([List of parameter values]2, label2), etc...]
def make_model(tuple_list):

	# a) creating a LogReg classifier
	classifierLR = nltk.classify.SklearnClassifier(LogisticRegression())
	classifierLR.train(tuple_list)

	return classifierLR

def param_tuning(features, labels):
	
	classifier = nltk.classify.SklearnClassifier(LogisticRegression())
	#create hyperparameter search space
	penalty = ['l1','l2']
	C = np.logspace(0,4,10)
	hyperparameters = dict(C=C, penalty=penalty)

	#create + do grid search
	gridsearch = GridSearchCV(model,hyperparameters,cv=5,verbose=0)
	best_model = gridsearch.fit(features, labels)

	return best_model.predict(features)
