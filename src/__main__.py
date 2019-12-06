from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
from src.pre_processing.preProcess import CSV2Numpy
import src.feature_engineering.feature_engineering as feature_eng
import random
import src.experiments.experiments as experiments
from sklearn.base import clone

# Preprocess the data
# Get the positive and negative datasets
positive,negative = CSV2Numpy()

# apply log transformation
positive = feature_eng.log_transformation(positive)
negative = feature_eng.log_transformation(negative)


# Combine both the positive and negative data and then shuffle the data
dataSet = np.concatenate((positive,negative), 0)
np.random.shuffle(dataSet)

# Augment the feature matrix
interactions = feature_eng.first_order_interactions(dataSet)
dataSet = feature_eng.augment_data(dataSet,interactions)

# 70% training, 30% test
trainingData = dataSet[:int(len(dataSet)*0.7), :]
testingData = dataSet[int(len(dataSet) * 0.7):, :]

#our training data and label
x_train = trainingData[:, :-1]
y_train = trainingData[:, -1]
y_train = y_train.astype('int')

#our testing data and label
x_test = testingData[:, :-1]
y_test = testingData[:, -1]
y_test = y_test.astype('int')

# Create the model(s)
logReg = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=1000)
svm = SVC()
# Create logReg model for AdaBoost
logReg2 = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=1000)
ada = AdaBoostClassifier(base_estimator=logReg2, n_estimators=1000)

print('='*20, clone(logReg).max_iter)

# Split into training and evaluation

# Train the models
# Assume x_train, y_train, x_test, y_test
logReg.fit(x_train,y_train)
svm.fit(x_train,y_train)
ada.fit(x_train, y_train)

# Evaluate the models using the validation set
# y_predict = logReg.predict(x_test)
# y_predict2 = logReg.predict(x_train)
y_predict = logReg.predict(x_test)
y_predict2 = logReg.predict(x_train)
y_ada = ada.predict(x_test)

print("="*20 + " RANDOM BASELINE " + "="*20)
y_random = np.array([random.randint(0,1) for y in y_test])

print("Macro: " + str(f1_score(y_test,y_random,average='macro')))
print("Micro: " + str(f1_score(y_test,y_random,average='micro')))
print("Accuracy: " + str(accuracy_score(y_test,y_random)))
print("ROC: " + str(roc_auc_score(y_test,y_random)))

print("="*20 + " LOGISTIC REGRESSION " + "="*20)
print("Macro: " + str(f1_score(y_test,y_predict,average='macro')))
print("Micro: " + str(f1_score(y_test,y_predict,average='micro')))
print("Accuracy: " + str(logReg.score(x_test,y_test)))
print("ROC: " + str(roc_auc_score(y_test,y_predict)))

print("="*20 + " ADA BOOST ON LOGISTIC REGRESSSION " + "="*20)
print("Macro: " + str(f1_score(y_test,y_ada,average='macro')))
print("Micro: " + str(f1_score(y_test,y_ada,average='micro')))
print("Accuracy: " + str(accuracy_score(y_test,y_ada)))
print("ROC: " + str(roc_auc_score(y_test,y_ada)))

print("="*20 + " BELOW IS ON TRAINING SET " + "="*20)

print("Macro: " + str(f1_score(y_train,y_predict2,average='macro')))
print("Micro: " + str(f1_score(y_train,y_predict2,average='micro')))
print("Accuracy: " + str(logReg.score(x_train,y_train)))
print("ROC: " + str(roc_auc_score(y_train,y_predict2)))

######################## Try K-Folds ######################

# Create the model(s)
logReg = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced')
svm = SVC()
# Create logReg model for AdaBoost
logReg2 = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=300)
ada = AdaBoostClassifier(base_estimator=logReg2, n_estimators=1000)

best_classifier = experiments.k_fold(classifiers = [logReg, svm, ada],dataset = dataSet, k = 5)