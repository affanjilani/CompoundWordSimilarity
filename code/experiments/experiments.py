import sklearn.model_selection as sk_model_sel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.base import clone
import numpy as np
from tqdm import tqdm

# Perform a fold on a given classifier, given the training and test data. Return the metric of choice on this data
def perform_fold(classifier, train_data, test_data, metric='ROC'):

    #Fit the classifier on the training data
    classifier.fit(train_data[0], train_data[1])

    #Predict the labels of the test data
    y_predicted = classifier.predict(test_data[0])
    
    if metric == 'ROC':
        return roc_auc_score(test_data[1],y_predicted)

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

            
if __name__ == "__main__":
    pass
