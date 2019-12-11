from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from src.pre_processing.preProcess import preProcess_pipeline
from src.feature_engineering.feature_engineering import feature_engineering_pipeline
from src.experiments.experiments import experiment_pipeline

# Create the model(s)
logReg = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=1000)
svm = SVC()
# Create logReg model for AdaBoost
logReg2 = LogisticRegression(solver='lbfgs', penalty='none',class_weight='balanced', max_iter=1000)
ada = AdaBoostClassifier(base_estimator=logReg2, n_estimators=1000)

# Get the original shuffled Dataset
dataSet = preProcess_pipeline()

# next, pass the data set to feature engineering pipeline
dataSet = feature_engineering_pipeline(dataSet,'SF',['i','l'], True)

# finally run our final dataset through the experiment pipeline
experiment_pipeline([logReg,svm,ada], dataSet, 5, 'roc', True, 0.9)