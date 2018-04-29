from tec.ic.ia.p1.models.Model import Model
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

class SupportVectorMachine(Model):
    def __init__(self, samples_train, samples_test, prefix):
        super().__init__(samples_train, samples_test, prefix)

    def execute(self):
        scaler = StandardScaler()
        samples_train = self.samples_train[0]
        samples_train = scaler.fit_transform(samples_train)
        samples_test = self.samples_test[0]
        samples_test = scaler.fit_transform(samples_test)

        # Create the SVC model object
        C = 1.0 # SVM regularization parameter
        svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr')
        svc.fit(samples_train, self.samples_train[1])
        predicted = svc.predict(samples_test)

        # get the accuracy
        print ("\nAccuracy: ", accuracy_score(self.samples_test[1], predicted))

        # Create the SVC model object
        # C = 1.0 # SVM regularization parameter
        svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr')

        svc.fit(samples_train, self.samples_train[1])
        predicted = svc.predict(samples_test)
        print ("\nAccuracy: ", accuracy_score(self.samples_test[1], predicted))
