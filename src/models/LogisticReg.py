from sklearn.linear_model import LogisticRegression

class LogisticRegGridSearch:
    def __init__(self):
        pass

    # Returns list of the different models
    def gen_models():
        models = []
        penalty = ['none','l1','l2']
        C = [1e3,1e2,1e1,1,10,100]
        cw = [None,'balanced']
        t = [1e5,1e4,1e3,1e2]

        for p in penalty:
            for c in C:
                for class_weight in cw:
                    for tol in t:
                        models.append(LogisticRegression(penalty=penalty, C=c, class_weight=class_weight, tol=tol))
        return models




