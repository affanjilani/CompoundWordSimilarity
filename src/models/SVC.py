from sklearn.svm import SVC

# method that will returna list of models as objects
def generate_models_SVC():
    # list of possible hyperparameters
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degrees = ['1', '2', '3']
    gammas = ['scale','auto']

    listOfModels = []
    # iterate through every possible hyper parameter
    for c in Cs:
        for kernel in kernels:
            for degree in degrees:
                for gamma in gammas:
                    # create the model and append to list of models
                    model = SVC(C=c, kernel=kernel,degree=degree,gamma=gamma)
                    listOfModels.append(model)
    # finally, return the list of models
    return listOfModels
