from models.Model import Model


class DecisionTree(Model):
    def __init__(self, samples, prefix, pruning_threshold):
        super().__init__(samples, prefix)
        self.pruning_threshold = pruning_threshold

    def execute(self):
        pass
