from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

def gen_ada_models(estimator = None):
    models = []

    n_estimators = [50,100,200]

    learning_rate = [1e2,1e1,1.,2.,10.]

    for num_est in n_estimators:
        for l_r in learning_rate:
            models.append(AdaBoostClassifier(n_estimators=num_est,learning_rate=l_r))
             if estimator is not None:
                 models.append(AdaBoostClassifier(base_estimator=estimator,learning_rate=l_r, n_estimators=num_est))

    return models
   

