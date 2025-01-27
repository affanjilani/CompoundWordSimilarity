from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from src.pre_processing.preProcess import preProcess_pipeline
from src.feature_engineering.feature_engineering import generate_datasets
from src.experiments.experiments import experiment_pipeline, random_labelling
from src.models.GridSearch import grid_search
from tqdm import tqdm
import random
from sklearn.metrics import f1_score, accuracy_score,roc_auc_score
import numpy as np

# # Create the model(s)
# logReg = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=2000)
# svm = SVC(gamma='auto')
# # Create logReg model for AdaBoost
# logReg2 = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=2000)
# ada = AdaBoostClassifier(base_estimator=logReg2, n_estimators=1000)
#
# # Get the original shuffled Dataset
# dataSet = preProcess_pipeline()
#
# # next, pass the data set to feature engineering pipeline
# dataSet = feature_engineering_pipeline(dataSet,'SF',['bc','l','i'], True, 0.12)
#
# # finally run our final dataset through the experiment pipeline
# experiment_pipeline([logReg,svm,ada], dataSet, 10, 'roc', True, 0.9)



# generate the combination of datasets
dataSets = generate_datasets()

# iterate through each dataset
datasetnum = 1
for dataset in tqdm(dataSets):
    print("="*20,datasetnum,"="*20)

<<<<<<< HEAD
    testingData = dataset[int(len(dataset)*0.9):,:]
    y_test = testingData[:, -1]
    y_test = y_test.astype('int')

    finalRandArray = []
    for i in range(10):
        y_predicted = []
        for y in y_test:
            y_random = random.randint(0,1)
            y_predicted.append(y_random)

        y_predicted = np.array(y_predicted)

        finalRandArray.append(f1_score(y_test,y_predicted,average='macro'))
    print(np.average(np.array(finalRandArray)))
    # print(roc_auc_score(y_test,y_predicted))
    # LRModels = grid_search('lr')
    # ADAModels = grid_search('ada')
    # NBModels = grid_search('nb')
    # SVCModels = grid_search('svc')
    #
    # experiment_pipeline(LRModels, dataset, 5, 'macro', True, 0.9)
    # experiment_pipeline(ADAModels, dataset, 5, 'macro', True, 0.9)
    # experiment_pipeline(SVCModels, dataset, 5, 'macro', True, 0.9)
=======
    LRModels = grid_search('lr')
    ADAModels = grid_search('ada')
    NBModels = grid_search('nb')
    SVCModels = grid_search('svc')

    experiment_pipeline(LRModels, dataset, 5, 'macro', True, 0.9)
    experiment_pipeline(ADAModels, dataset, 5, 'macro', True, 0.9)
    experiment_pipeline(SVCModels, dataset, 5, 'macro', True, 0.9)
    random_labelling(dataset)

>>>>>>> 9c2216136bcbb3cb7c5462f21a22577421149086
    datasetnum +=1

