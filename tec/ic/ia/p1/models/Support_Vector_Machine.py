from tec.ic.ia.p1.models.Model import Model
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class SupportVectorMachine(Model):
    def __init__(self, samples_train, samples_test, prefix):
        super().__init__(samples_train, samples_test, prefix)

    def execute(self):
        scaler = StandardScaler()
        samples_train = self.samples_train[0]
        samples_train = scaler.fit_transform(samples_train)
        samples_test = self.samples_test[0]
        samples_test = scaler.fit_transform(samples_test)

        parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
        clf.fit(samples_train, self.samples_train[1])

        print(clf.best_params_)
        print()
        print("Grid scores on training set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        # # Create the SVC model object
        # C = 1.0 # SVM regularization parameter
        # svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovo')
        # svc.fit(samples_train, self.samples_train[1])
        # predicted = svc.predict(samples_test)
        #
        # # get the accuracy
        # print ("\nAccuracy: ", accuracy_score(self.samples_test[1], predicted))
        #
        # # Create the SVC model object
        # # C = 1.0 # SVM regularization parameter
        # svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovo')
        # svc.fit(samples_train, self.samples_train[1])
        # predicted = svc.predict(samples_test)
        # print ("\nAccuracy: ", accuracy_score(self.samples_test[1], predicted))
        #
        #
        # svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr')
        # svc.fit(samples_train, self.samples_train[1])
        # predicted = svc.predict(samples_test)
        # print ("\nAccuracy: ", accuracy_score(self.samples_test[1], predicted))
        #
        # svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr')
        # svc.fit(samples_train, self.samples_train[1])
        # predicted = svc.predict(samples_test)
        # print ("\nAccuracy: ", accuracy_score(self.samples_test[1], predicted))
