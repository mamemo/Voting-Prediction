from abc import ABC, abstractmethod

"""
This class implement an abstract class as a definition of a model.
"""

class Model(ABC):
    def __init__(self, samples_train, samples_test, prefix):
        self.samples_train = samples_train
        self.samples_test = samples_test
        self.prefix = prefix

    @abstractmethod
    def execute(self):
        pass
