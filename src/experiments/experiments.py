import sklearn.model_selection as sk_model_sel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.base import clone
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from src.pre_processing.preProcess import CSV2Numpy

# Perform a fold on a given classifier, given the training and test data. Return the metric of choice on this data
def perform_fold(classifier, train_data, test_data, metric='ROC'):

    #Fit the classifier on the training data
    classifier.fit(train_data[0], train_data[1])

    #Predict the labels of the test data
    y_predicted = classifier.predict(test_data[0])
    
    if metric.lower() == 'roc':
        return roc_auc_score(test_data[1], y_predicted)
    elif metric.lower() == 'macro':
        return f1_score(test_data[1], y_predicted, average='macro')
    elif metric.lower() == 'micro':
        return f1_score(test_data[1], y_predicted, average='micro')
    elif metric.lower() == 'acc':
        return accuracy_score(test_data[1], y_predicted)

# The K-folds routine. Input are a list of initialized classifiers, as well as the dataset.
# The dataset should be a nx(m+1) where n = number of samples, m = number of features, and the last
# column are the labels. Prints the scores for all classifiers and returns the best one.
def k_fold(classifiers, dataset, k = 5, metric = 'ROC', verbose = True):
    kf = sk_model_sel.KFold(n_splits=k, shuffle=True)

    # Initialize dict
    scores = {}
    for clf in classifiers:
        scores[type(clf).__name__] = np.array([])

    # For each split
    for train_index, test_index in tqdm(kf.split(dataset)):
        train = dataset[train_index]
        x_train = train[:,:-1]
        y_train = train[:,-1].astype('int')

        test = dataset[test_index]
        x_test = test[:,:-1]
        y_test = test[:,-1].astype('int')

        # We send this into each classifier and perform a fold
        for clf in tqdm(classifiers):
            clf_name = type(clf).__name__

            # Get the score on this split
            score = perform_fold(clone(clf),(x_train,y_train),(x_test,y_test), metric=metric)

            # Add score to the dict
            scores[clf_name] = np.append(scores[clf_name], score)

    # We print out the classifier scores and then return the best performing classifier
    best_clf = ('',0)

    if verbose:
        print('='*20, metric, '='*20)

    for clf in classifiers:
        clf_name = type(clf).__name__

        # Get the average score for the clf
        clf_score = np.average(scores[clf_name])

        if verbose:
            print(clf_name,':',clf_score)

        if best_clf[1] < clf_score:
            best_clf = (clf,clf_score)

    if verbose:
        print("="*20, 'Best Classifier', type(best_clf[0]).__name__, metric, ':',best_clf[1])
    return clone(best_clf[0])

## Method that does the entire experiment by first running k fold to get the best model then does the training and testing
def experiment_pipeline(classifiers, dataSet, k = 5, metric = 'ROC', verbose=True, split = 0.7):
    # first thing's first, get the best model by running k fold
    model = k_fold(classifiers, dataSet, k, metric, False)

    # now that we have the best model, split the dataset based on the datasplit

    # We first shuffle the data set
    np.random.shuffle(dataSet)

    # default: 70% training, 30% test
    trainingData = dataSet[:int(len(dataSet) * split), :]
    testingData = dataSet[int(len(dataSet) * split):, :]

    # our training data and label
    x_train = trainingData[:, :-1]
    y_train = trainingData[:, -1]
    y_train = y_train.astype('int')

    # our testing data and label
    x_test = testingData[:, :-1]
    y_test = testingData[:, -1]
    y_test = y_test.astype('int')

    # Train our model with the training set
    model.fit(x_train,y_train)

    # Evaluate the models using the validation set
    y_predict = model.predict(x_test)

    # Print the output based on the requested metric
    if verbose:
        print('=' * 20, metric, '=' * 20)

    if metric.lower() == 'roc':
        print(type(model).__name__ + ": "+ str(roc_auc_score(y_test,y_predict)))
    elif metric.lower() == 'macro':
        print(type(model).__name__ + ": "+ str(f1_score(y_test,y_predict,average='macro')))
    elif metric.lower() == 'micro':
        print(type(model).__name__ + ": "+ str(f1_score(y_test,y_predict,average='micro')))
    elif metric.lower() == 'acc':
        print(type(model).__name__ + ": "+ str(accuracy_score(y_test,y_predict)))

if __name__ == "__main__":
    # LR = LogisticRegression()
    # pos,neg = CSV2Numpy()
    # dataSet = np.concatenate((pos, neg), 0)
    # experiment_pipeline([LR], dataSet, 10, 'ROC', True, 0.9)
    pass
