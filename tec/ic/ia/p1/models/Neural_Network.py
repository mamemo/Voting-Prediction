from tec.ic.ia.p1.models.Model import Model
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import ast


class NeuralNetwork(Model):
    def __init__(self, samples_train, samples_test, prefix, layers, units_per_layer, activation_function):
        super().__init__(samples_train, samples_test, prefix)
        self.layers = layers
        self.units_per_layer = ast.literal_eval(units_per_layer)
        self.activation_function = activation_function
        # HyperParameters
        self.training_epochs = 3000
        self.batch_size = 500

    def execute(self):
        dim_input = self.samples_train[0].shape[1]
        dim_output = self.samples_train[1].shape[1]

        # Create the model
        model = Sequential()

        # Add layers
        model.add(Dense(dim_input, input_dim=dim_input,
                        kernel_initializer="uniform", activation='relu'))
        for i in range(self.layers):
            model.add(Dense(
                self.units_per_layer[i], kernel_initializer="uniform", activation=self.activation_function))
        model.add(Dense(dim_output, kernel_initializer="uniform", activation='softmax'))

        # Compile the model
        #optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        optimizer = optimizers.SGD(lr=0.01, momentum=0.1, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])

        # Fit the model
        model.fit(self.samples_train[0], self.samples_train[1], validation_data=(
            self.samples_test[0], self.samples_test[1]), nb_epoch=self.training_epochs, batch_size=self.batch_size)

        # Evaluate the model
        scores = model.evaluate(self.samples_test[0], self.samples_test[1])
        print ("Accuracy: %.2f%%" %(scores[1]*100))
