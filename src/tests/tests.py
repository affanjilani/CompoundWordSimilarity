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

def vanilla_dataset():
    positive,negative = CSV2Numpy()
    dataSet = np.concatenate((positive,negative), 0)
    np.random.seed(42)
    np.random.shuffle(dataSet)

    return dataSet

def log_dataset():
    log_augmentation = feature_eng.log_transformation(vanilla_dataset(),None)
    dataset = vanilla_dataset()
    dataset = feature_eng.augment_data(dataset,log_augmentation)

    return dataset

def random_features_dataset():
    dataset = vanilla_dataset()
    random_features = np.random.randn(np.shape(dataset)[0],10)
    dataset = feature_eng.augment_data(dataset,random_features)

    return dataset

def random_entries_dataset():
    dataset = vanilla_dataset()
    random_features = np.random.randn(np.shape(dataset)[0],np.shape(dataset)[1]-1)

    random_labels = np.array([random.randint(0,1) for y in range(np.shape(dataset)[0])]).reshape(-1,1)

    random_full = np.hstack((random_features,random_labels))

    return np.concatenate((dataset,random_full),0)

# Splits a dataset and returns (x_train,y_train,x_test,y_test)
def splitData(dataset, split = 0.9):
    dataset = np.array(dataset)
    train = dataset[:int(np.shape(dataset)[0] * split),:]
    test = dataset[int(np.shape(dataset)[0] * split):,:]

    return (train[:,:-1],train[:,-1].astype('int'),test[:,:-1],test[:,-1].astype('int'))

def init_models():
    logReg = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=1000)
    svm = SVC(gamma='scale')
    # Create logReg model for AdaBoost
    logReg2 = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=1000)
    ada = AdaBoostClassifier(base_estimator=logReg2, n_estimators=1000)

    return [logReg,svm,ada]

def print_stats(dataset, models, split = 0.9):
    x_train, y_train, x_test, y_test = splitData(dataset)
    for model in models:
        # Fit
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)

        print('\t\t',"#"*20 + type(model).__name__ + "#"*20)
        print('\t\t',"Macro: " + str(f1_score(y_test,y_predict,average='macro')))
        print('\t\t',"Micro: " + str(f1_score(y_test,y_predict,average='micro')))
        print('\t\t',"Accuracy: " + str(accuracy_score(y_test,y_predict)))
        print('\t\t',"ROC: " + str(roc_auc_score(y_test,y_predict)))


##################### VANILLA #######################
vanillaDataset = vanilla_dataset()
print('Vanilla shape',np.shape(vanillaDataset))
x_train, y_train, x_test, y_test = splitData(vanillaDataset)
print('vanilla train', np.shape(x_train))

datasets = [('Vanilla',vanilla_dataset()),('LOG',log_dataset()),('Random Features',random_features_dataset()),('Random Entires',random_entries_dataset())]

for split in [0.1,0.9]:
    print("*"*20,'Split:',split,'*'*20)

    for name, dataset in datasets:
        print('\t',"="*20,'Dataset',name,"="*20)

        print_stats(dataset,init_models(),split=split)



