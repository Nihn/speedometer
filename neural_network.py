from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from threading import Thread


class NeuralNetwork():

    def __init__(self, network_tuple, epochs=1):
        self.network = buildNetwork(*network_tuple)
        self.ds = SupervisedDataSet(2, 1)
        self.training = Thread(target=self.train, args=(epochs,))
        self.training.daemon = True
        self.done = False

    def add_sample(self, x, y, result):
        self.ds.addSample((x, y), (result,))

    def train(self, epochs=1):
        trainer = BackpropTrainer(self.network, self.ds)
        trainer.trainEpochs(epochs)
        self.done = True

    def is_done(self):
        if self.done:
            self.done = False
            return True
        return False

    def result(self, x, y):
        return self.network.activate((x, y))
