from tec.ic.ia.p1.models.Model import Model


class DecisionTree(Model):
    def __init__(self, samples_train, samples_test, prefix, pruning_threshold):
        super().__init__(samples_train, samples_test, prefix)
        self.pruning_threshold = pruning_threshold

    def execute(self):
        pass
