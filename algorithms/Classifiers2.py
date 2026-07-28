import optuna
import pandas
import xgboost
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text


class CLF(object):
    def __init__(self, classifier):
        self.clf = classifier
        self.clfmodel = None

    def getCLF(self):
        if self.clf == 'KNN':
            self.clfmodel = KNeighborsClassifier(n_jobs=-1)
        elif self.clf == 'NB':
            self.clfmodel = GaussianNB()
        elif self.clf == 'LR':
            self.clfmodel = LogisticRegression(n_jobs=-1)
        elif self.clf == 'DT':
            self.clfmodel = DecisionTreeClassifier()
        elif self.clf == 'RF':
            self.clfmodel = RandomForestClassifier(n_jobs=-1)
        elif self.clf == 'SVM':
            self.clfmodel = SVC()
        # elif self.clf == 'linearSVM':
        #     self.clfmodel = LinearSVC()
        elif self.clf == 'linearSVM':
            self.clfmodel = LinearSVC(
                penalty='l2',
                loss='squared_hinge',
                dual=True
            )
        elif self.clf == 'MLP':
            self.clfmodel = MLPClassifier()
        elif self.clf == 'Bagging':
            self.clfmodel = BaggingClassifier(n_jobs=-1)
        elif self.clf == 'AdaBoost':
            self.clfmodel = AdaBoostClassifier()
        elif self.clf == 'GBM':
            self.clfmodel = GradientBoostingClassifier()
        elif self.clf == 'XGBoost':
            self.clfmodel = xgboost.XGBClassifier()

        return self.clfmodel


class OptimizingCLF(object):
    def __init__(self, trainx, trainy, testx, testy, classifier):
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy
        self.clf = classifier

        self.clfmodel = None

    def optTree_fun(self, trial):
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, len(self.trainy))
        clfmodel = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

        # training model
        clfmodel.fit(self.trainx, self.trainy)

        # predict
        y_pred = clfmodel.predict(self.trainx)

        return matthews_corrcoef(self.trainy, y_pred)

    def getOptTree(self):
        study = optuna.create_study(study_name='classifier_tuning', load_if_exists=False, directions=['maximize'],
                                    sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: self.optTree_fun(trial),
                       n_trials=20, n_jobs=-1)  # callbacks=[logging_callback],

        params = study.best_params

        return DecisionTreeClassifier(**params)

    def optCLF_fun(self, trial):
        if self.clf == 'KNN':
            # KNN: neighbors from 1 to sqrt(n)
            # n = np.ceil(np.sqrt(len(self.trainy)))
            n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            p = trial.suggest_int('p', 1, 5)
            self.clfmodel = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                 weights=weights,
                                                 p=p,
                                                 n_jobs=-1)

        elif self.clf == 'NB':
            NBType = trial.suggest_categorical('NBType', ['GaussianNB', 'BernoulliNB'])  # 'MultinomialNB',
            if NBType == 'GaussianNB':
                self.clfmodel = GaussianNB()
            # elif NBType == 'MultinomialNB':
            #     clfmodel = MultinomialNB()
            elif NBType == 'BernoulliNB':
                self.clfmodel = BernoulliNB()

        elif self.clf == 'LR':
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            C = trial.suggest_float('C', 0.0001, 1000)
            tol = trial.suggest_float('tol', 0.00001, 1)
            self.clfmodel = LogisticRegression(penalty=penalty, C=C, tol=tol, solver='liblinear')

        elif self.clf == 'DT':
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            max_depth = trial.suggest_int('max_depth', 1, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, len(self.trainy))
            # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, len(train_label))
            self.clfmodel = DecisionTreeClassifier(criterion=criterion,
                                                   max_depth=max_depth,
                                                   min_samples_split=min_samples_split)

        elif self.clf == 'RF':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_int('max_depth', 1, 20)
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            self.clfmodel = RandomForestClassifier(n_estimators=n_estimators,
                                                   max_depth=max_depth,
                                                   criterion=criterion,
                                                   n_jobs=-1)

        elif self.clf == 'SVM':
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            C = trial.suggest_float('C', 0.001, 1000)
            if kernel == 'poly':
                degree = trial.suggest_int('degree', 1, 5)
                gamma = trial.suggest_float('gamma', 0.01, 100)
                coef0 = trial.suggest_float('coef0', 0, 10)
                self.clfmodel = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0
                                    )  # max_iter=100000
            elif kernel == 'rbf':
                gamma = trial.suggest_float('gamma', 0.01, 100)
                self.clfmodel = SVC(kernel=kernel, C=C, gamma=gamma)
            elif kernel == 'sigmoid':
                gamma = trial.suggest_float('gamma', 0.01, 100)
                self.clfmodel = SVC(kernel=kernel, C=C, gamma=gamma)
            else:
                self.clfmodel = SVC(kernel=kernel, C=C)

        elif self.clf == 'linearSVM':
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            # loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])
            C = trial.suggest_float('C', 0.001, 1000)
            # self.clfmodel = LinearSVC(penalty=penalty, C=C)

            if penalty == 'l1':
                self.clfmodel = LinearSVC(
                    penalty='l1',
                    loss='squared_hinge',
                    dual=False,  # ⭐关键
                    C=C
                )
            else:
                self.clfmodel = LinearSVC(
                    penalty='l2',
                    loss='squared_hinge',
                    dual=True,
                    C=C
                )


        elif self.clf == 'MLP':
            n_neurons = trial.suggest_int('n_neurons', 50, 500)
            n_layers = trial.suggest_int('n_layers', 1, 5)
            hidden_layer_sizes = (n_neurons,) * n_layers
            activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
            alpha = trial.suggest_float('alpha', 0.00001, 1)
            # max_iter = trial.suggest_int('max_iter', 50, 500)
            # learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
            self.clfmodel = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                          activation=activation,
                                          alpha=alpha)

        elif self.clf == 'Bagging':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_samples = trial.suggest_float('max_samples', 0.1, 1)
            # max_features = trial.suggest_int('max_features', 0.1, 1)
            self.clfmodel = BaggingClassifier(n_estimators=n_estimators,
                                              max_samples=max_samples,
                                              n_jobs=-1)

        elif self.clf == 'AdaBoost':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 10)
            self.clfmodel = AdaBoostClassifier(n_estimators=n_estimators,
                                               learning_rate=learning_rate)

        elif self.clf == 'GBM':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 10)
            max_depth = trial.suggest_int('max_depth', 1, 20)
            # min_samples_split = trial.suggest_int('min_samples_split', 2, len(self.trainy))
            # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, len(train_label))
            # max_features = trial.suggest_int('max_features', 1, train_data.shape[1])
            # subsample = trial.suggest_float('subsample', 0.1, 1)
            self.clfmodel = GradientBoostingClassifier(n_estimators=n_estimators,
                                                       learning_rate=learning_rate,
                                                       max_depth=max_depth)
        elif self.clf == 'XGBoost':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            eta = trial.suggest_float('eta', 0, 1)
            max_depth = trial.suggest_int('max_depth', 1, 20)
            # subsample = trial.suggest_float('subsample', 0.5, 1)
            # colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1)
            # gamma = trial.suggest_float('gamma', 0, 10)
            # min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            # reg_lambda = trial.suggest_float('reg_lambda', 0.1, 10)
            # reg_alpha = trial.suggest_float('reg_alpha', 0.1, 10)
            # max_delta_step = trial.suggest_int('max_delta_step', 0, 10)
            # self.clfmodel = xgboost.XGBClassifier(n_estimators=n_estimators, eta=eta, max_depth=max_depth,
            #                                subsample = subsample, colsample_bytree = colsample_bytree,
            #                                gamma=gamma, min_child_weight=min_child_weight,
            #                                reg_lambda=reg_lambda, reg_alpha=reg_alpha
            #                                max_delta_step=max_delta_step,)
            self.clfmodel = xgboost.XGBClassifier(n_estimators=n_estimators,
                                                  eta=eta,
                                                  max_depth=max_depth
                                                  )

        # training model
        model = self.clfmodel
        # self.clfmodel.fit(self.trainx, self.trainy)
        model.fit(self.trainx, self.trainy)

        # predict
        # y_pred = self.clfmodel.predict(self.testx)
        y_pred = model.predict(self.testx)

        return matthews_corrcoef(self.testy, y_pred)

    def getOptCLF(self):
        study = optuna.create_study(study_name='classifier_tuning', load_if_exists=False, directions=['maximize'],
                                    sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: self.optCLF_fun(trial), n_trials=20, n_jobs=-1)  # callbacks=[logging_callback],

        params = study.best_params

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
            return SVC(**params)

        if self.clf == 'linearSVM':
            penalty = params.get('penalty')
            C = params.get('C')

            if penalty == 'l1':
                return LinearSVC(
                    penalty='l1',
                    loss='squared_hinge',
                    dual=False,  # ⭐必须
                    C=C
                )
            else:
                return LinearSVC(
                    penalty='l2',
                    loss='squared_hinge',
                    dual=True,
                    C=C
                )
            # return LinearSVC(**params)


        if self.clf == 'Bagging':
            return BaggingClassifier(**params, n_jobs=-1)

        if self.clf == 'AdaBoost':
            return AdaBoostClassifier(**params)

        if self.clf == 'GBM':
            return GradientBoostingClassifier(**params)

        if self.clf == 'XGBoost':
            return  xgboost.XGBClassifier(**params)

    def get_interpretability_report(self, feature_names):
        """
            Generate interpretability analysis report

            Returns:
                report: Dictionary containing interpretability metrics
        """

        if self.clfmodel is None:
            raise ValueError("Please call fit_predict() first to train the model")

        # sort feature importance
        if self.clf in ['DT', 'RF', 'GBM', 'XGBoost']:
            feature_importances_ = self.clfmodel.feature_importances_
        else: # ['LR', 'linearSVM']
            feature_importances_ = self.clfmodel.coef_.T

        feature_importances_ = pandas.DataFrame(feature_importances_)
        feature_importances_.index = feature_names
        # feature_importances_.columns = feature_names
        sorted_feature_importances = feature_importances_.sort_values(ascending=False, by=0)

        if self.clf == 'DT':
            # Tree complexity metrics
            n_nodes = self.clfmodel.tree_.node_count
            max_depth = self.clfmodel.get_depth()

            # select top-k features for output tree rules
            # top_importances = sorted_feature_importances[sorted_feature_importances[0] >= threshold]
            # top_features = top_importances.index
            #
            # mask = [fea in top_features for fea in feature_names]
            # custom_names = [feature_names[i] if m else "ignore" for i, m in enumerate(mask)]
            # tree_rules = export_text(self.tree, feature_names=custom_names)

            # 在调用 export_text 之前添加转换
            if feature_names is not None:
                # 确保 feature_names 是列表而不是numpy数组
                feature_names = list(feature_names)
            tree_rules = export_text(self.clfmodel, feature_names=feature_names)

            report = {
                'tree': self.clfmodel,
                'feature_importances': sorted_feature_importances,
                'tree_max_depth': max_depth,
                'tree_n_nodes': n_nodes,
                'tree_rules': tree_rules
            }
        elif self.clf in ['RF', 'GBM', 'XGBoost']:
            report = {
                'tree': self.clfmodel,
                'feature_importances': sorted_feature_importances,
            }
        elif self.clf in ['LR', 'linearSVM']:
            report = {
                'model': self.clfmodel,
                'feature_importances': sorted_feature_importances,
            }

        return report
