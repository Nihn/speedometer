from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

from threading import Thread


class NeuralNetwork(object):

    def __init__(self, network_tuple=None, epochs=1, save='', load=''):

        if not network_tuple and not load:
            raise TypeError('Network tupe or load must be provided.')

        self.network = NetworkReader.readFrom(load) if load else \
            buildNetwork(*network_tuple)
        self.ds = SupervisedDataSet(inp=2, target=1)
        self.training = Thread(target=self.train, args=(epochs,))
        self.training.daemon = True
        self.done = False
        self.save = save

    def add_sample(self, x, y, result):
        self.ds.addSample((x, y), (result,))

    def train(self, epochs=1):
        self.done = False
        trainer = BackpropTrainer(self.network, self.ds)
        trainer.trainEpochs(epochs)
        self.done = True
        if self.save:
            NetworkWriter.writeToFile(self.network, self.save)

    def is_done(self):
        return self.done

    def result(self, x, y):
        return self.network.activate((x, y))
