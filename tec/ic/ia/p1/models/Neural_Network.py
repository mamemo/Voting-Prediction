from tec.ic.ia.p1.models.Model import Model


class NeuralNetwork(Model):
    def __init__(self, samples_train, samples_test, prefix, layers, units_per_layer, activation_function):
        super().__init__(samples_train, samples_test, prefix)
        self.layers = layers
        self.units_per_layer = units_per_layer
        self.activation_function = activation_function

    def execute(self):
        pass
