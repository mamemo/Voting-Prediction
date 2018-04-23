from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, samples, prefix):
        self.samples = samples
        self.prefix = prefix

    @abstractmethod
    def execute(self):
        pass