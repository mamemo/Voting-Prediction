from models.Model import Model


class KNearestNeighbors(Model):
    def __init__(self, samples, prefix, k):
        super().__init__(samples, prefix)
        self.k = k

    def execute(self):
        pass
