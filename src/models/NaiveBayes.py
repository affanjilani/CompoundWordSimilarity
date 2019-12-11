from sklearn.naive_bayes import MultinomialNB

# method that will returna list of models as objects
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
