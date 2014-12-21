from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from threading import Thread


class NeuralNetwork(object):

    def __init__(self, network_tuple, epochs=1):
        self.network = buildNetwork(*network_tuple)
        self.ds = SupervisedDataSet(inp=2, target=1)
        self.training = Thread(target=self.train, args=(epochs,))
        self.training.daemon = True
        self.done = False

    def add_sample(self, x, y, result):
        self.ds.addSample((x, y), (result,))

    def train(self, epochs=1):
        self.done = False
        trainer = BackpropTrainer(self.network, self.ds)
        trainer.trainEpochs(epochs)
        self.done = True

    def is_done(self):
        return self.done

    def result(self, x, y):
        return self.network.activate((x, y))
