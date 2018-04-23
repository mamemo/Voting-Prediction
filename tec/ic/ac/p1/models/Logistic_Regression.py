from tec.ic.ac.p1.models.Model import Model


class LogisticRegression(Model):
    def __init__(self, samples, prefix, regularization):
        super().__init__(samples, prefix)
        self.regularization = regularization

    def execute(self):
        pass
