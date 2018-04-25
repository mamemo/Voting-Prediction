from models.Model import Model


class NeuralNetwork(Model):
    def __init__(self, samples, prefix, layers, units_per_layer, activation_function):
        super().__init__(samples, prefix)
        self.layers = layers
        self.units_per_layer = units_per_layer
        self.activation_function = activation_function

    def execute(self):
        pass
