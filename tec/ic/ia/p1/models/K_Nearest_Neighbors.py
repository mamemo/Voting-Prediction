from tec.ic.ia.p1.models.Model import Model
import numpy as np

class KNearestNeighbors(Model):
    def __init__(self, samples_train, samples_test, prefix, k):
        super().__init__(samples_train, samples_test, prefix)
        self.k = k

    def execute(self):
        pass
