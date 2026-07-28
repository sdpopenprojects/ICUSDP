import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class CLF_paramers(object):
    def __init__(self, trainx, trainy, testx, testy, classifier):
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy
        self.clf = classifier

    def optCLF_fun(self, trial):
        if self.clf == 'KNN':
            # KNN: neighbors from 1 to sqrt(n)
            n = np.ceil(np.sqrt(len(self.trainy)))
            n_neighbors = trial.suggest_int('n_neighbors', 1, n)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            p = trial.suggest_int('p', 1, 5)
            clfmodel = KNeighborsClassifier(n_neighbors=n_neighbors,
                                            weights=weights,
                                            p=p, n_jobs=-1)

        elif self.clf == 'NB':
            NBType = trial.suggest_categorical('NBType', ['GaussianNB', 'BernoulliNB']) # 'MultinomialNB',
            if NBType == 'GaussianNB':
                clfmodel = GaussianNB()
            # elif NBType == 'MultinomialNB':
            #     clfmodel = MultinomialNB()
            elif NBType == 'BernoulliNB':
                clfmodel = BernoulliNB()

        elif self.clf == 'LR':
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            C = trial.suggest_float('C', 0.0001, 1000)
            tol = trial.suggest_float('tol', 0.00001, 1)
            clfmodel = LogisticRegression(penalty=penalty, C=C, tol=tol, solver='liblinear')

        elif self.clf == 'DT':
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            max_depth = trial.suggest_int('max_depth', 1, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, len(self.trainy))
            # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, len(train_label))
            clfmodel = DecisionTreeClassifier(criterion=criterion,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split)

        elif self.clf == 'RF':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_int('max_depth', 1, 20)
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            clfmodel = RandomForestClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth,
                                              criterion=criterion, n_jobs=-1)

        elif self.clf == 'SVM':
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            C = trial.suggest_float('C', 0.001, 1000)
            if kernel == 'poly':
                degree = trial.suggest_int('degree', 1, 5)
                gamma = trial.suggest_float('gamma', 0.01, 100)
                coef0 = trial.suggest_float('coef0', 0, 10)
                clfmodel = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0, max_iter=100000)
            elif kernel == 'rbf':
                gamma = trial.suggest_float('gamma', 0.01, 100)
                clfmodel = SVC(kernel=kernel, C=C, gamma=gamma, max_iter=100000)
            elif kernel == 'sigmoid':
                gamma = trial.suggest_float('gamma', 0.01, 100)
                clfmodel = SVC(kernel=kernel, C=C, gamma=gamma, max_iter=100000)
            else:
                clfmodel = SVC(kernel=kernel, C=C, max_iter=100000)

        elif self.clf == 'MLP':
            n_neurons = trial.suggest_int('n_neurons', 50, 500)
            n_layers = trial.suggest_int('n_layers', 1, 5)
            hidden_layer_sizes = (n_neurons,) * n_layers
            activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
            alpha = trial.suggest_float('alpha', 0.00001, 1)
            # max_iter = trial.suggest_int('max_iter', 50, 500)
            # learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
            clfmodel = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                     activation=activation,
                                     alpha=alpha)

        elif self.clf == 'Bagging':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_samples = trial.suggest_float('max_samples', 0.1, 1)
            # max_features = trial.suggest_int('max_features', 0.1, 1)
            clfmodel = BaggingClassifier(n_estimators=n_estimators,
                                         max_samples=max_samples,
                                         n_jobs=-1)

        elif self.clf == 'AdaBoost':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 10)
            clfmodel = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

        elif self.clf == 'GBM':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 10)
            max_depth = trial.suggest_int('max_depth', 1, 20)
            # min_samples_split = trial.suggest_int('min_samples_split', 2, len(self.trainy))
            # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, len(train_label))
            # max_features = trial.suggest_int('max_features', 1, train_data.shape[1])
            # subsample = trial.suggest_float('subsample', 0.1, 1)
            clfmodel = GradientBoostingClassifier(n_estimators=n_estimators,
                                                  learning_rate=learning_rate,
                                                  max_depth=max_depth
                                                  # subsample=subsample
                                                  )

        # training model
        clfmodel.fit(self.trainx, self.trainy)

        # predict
        y_pred = clfmodel.predict(self.testx)

        return matthews_corrcoef(self.testy, y_pred)

    def getCLF(self, params):
        if self.clf == 'KNN':
            return KNeighborsClassifier(**params, n_jobs=-1)

        if self.clf == 'NB':
            NBType = params.get('NBType')
            if NBType == 'GaussianNB':
                return GaussianNB()
            elif NBType == 'MultinomialNB':
                return MultinomialNB()
            elif NBType == 'BernoulliNB':
                return BernoulliNB()
        # if self.clf == 'NB':
        #     if params['NBType'] == 'GaussianNB':
        #         return GaussianNB()
        #     elif params['NBType'] == 'MultinomialNB':
        #         return MultinomialNB()
        #     elif params['NBType'] == 'BernoulliNB':
        #         return BernoulliNB()

        if self.clf == 'LR':
            return LogisticRegression(**params, solver='liblinear', n_jobs=-1)

        if self.clf == 'DT':
            return DecisionTreeClassifier(**params)

        if self.clf == 'RF':
            return RandomForestClassifier(**params, n_jobs=-1)

        if self.clf == 'MLP':
            n_neurons = params.get('n_neurons')
            n_layers = params.get('n_layers')
            hidden_layer_sizes = (n_neurons,) * n_layers
            params['hidden_layer_sizes'] = hidden_layer_sizes
            params.pop('n_neurons')
            params.pop('n_layers')
            return MLPClassifier(**params)

        if self.clf == 'SVM':
            return SVC(**params, max_iter=100000)

        if self.clf == 'Bagging':
            return BaggingClassifier(**params, n_jobs=-1)

        if self.clf == 'AdaBoost':
            return AdaBoostClassifier(**params)

        if self.clf == 'GBM':
            return GradientBoostingClassifier(**params)

