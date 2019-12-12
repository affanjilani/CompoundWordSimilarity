from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

# method that generates NB models
def generate_models_naive_bayes():
    listOfModels = []
    # iterate through the possible values of alpha as its the only hyperparameter
    for alpha in range(0, 101):
        # so we increment by 0.0
        alpha /= 100
        # create our model with the specified alpha
        model = MultinomialNB(alpha=alpha)
        # append the model
        listOfModels.append(model)
    # finally, return the list of models
    return listOfModels

# method that generates ada models
def generate_models_ada(estimator = None):
    models = []

    n_estimators = [50,100,200]

    learning_rate = [1e2,1e1,1.,2.,10.]

    for num_est in n_estimators:
        for l_r in learning_rate:
            models.append(AdaBoostClassifier(n_estimators=num_est,learning_rate=l_r))
            if estimator is not None:
                models.append(AdaBoostClassifier(base_estimator=estimator,learning_rate=l_r, n_estimators=num_est))

    return models

# method that generates SVC models
def generate_models_SVC():
    # list of possible hyperparameters
    # Cs = [0.01, 0.1, 1.0]
    # kernels = ['linear', 'poly', 'rbf']
    # degrees = ['2', '3', '4']
    # gammas = ['auto']

    Cs = [1.0, 2.0, 3.0]
    kernels = ['linear', 'rbf']
    gammas = ['auto']

    listOfModels = []
    # iterate through every possible hyper parameter
    for c in Cs:
        for kernel in kernels:
            for gamma in gammas:
                model = SVC(C=c, kernel=kernel, gamma=gamma)
                listOfModels.append(model)
    # finally, return the list of models
    return listOfModels


# model that generates logistic regression models
def generate_models_logReg():
    models = []
    penalty = ['none','l2']
    C = [1e3,1e2,1e1,1,10,100]
    cw = [None,'balanced']
    t = [1e5,1e4,1e3,1e2]

    for p in penalty:
        for c in C:
            for class_weight in cw:
                for tol in t:
                    models.append(LogisticRegression(solver='lbfgs', penalty=p, C=c, class_weight=class_weight, tol=tol,max_iter=2000))
    return models

# Method that generates models based on specific model
def grid_search(model):
    if model.lower() == 'nb':
        return generate_models_naive_bayes()
    elif model.lower() == 'svc':
        return generate_models_SVC()
    elif model.lower() == 'ada':
        return generate_models_ada()
    elif model.lower() == 'lr':
        return generate_models_logReg()