# Model Predictions
import nltk
import sklearn

def label(test_set, model):
	label = model.classify(test_set)
	