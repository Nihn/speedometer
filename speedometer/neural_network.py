from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

from numpy import random
from threading import Thread

from utils import scale


class NeuralNetwork(object):

    def __init__(self, network_tuple=None, epochs=1, save='', load='',
                 scale=1000, max_error=0):

        if not network_tuple and not load:
            raise TypeError('Network tuple or load must be provided.')

        self.network = NetworkReader.readFrom(load) if load else \
            buildNetwork(*network_tuple)
        self.ds = SupervisedDataSet(inp=2, target=1)
        self.scale = scale
        self.training = Thread(target=self.train, args=(epochs, max_error))
        self.training.daemon = True
        self.done = False
        self.save = save
        self.max_error = max_error
        self.epochs = epochs

    @scale
    def add_sample(self, y, dy, result):
        self.ds.addSample((y, dy), (result/100,))

    def train(self):

        self.done = False
        trainer = BackpropTrainer(self.network, learningrate=0.2)

        train_until_convergence(trainer=trainer, dataset=self.ds,
                                max_error=self.max_error,
                                max_epochs=self.epochs)

        self.done = True
        if self.save:
            NetworkWriter.writeToFile(self.network, self.save)

    def is_done(self):
        return self.done

    @scale
    def result(self, y, dy):
        return 100 * self.network.activate((y, dy))[0]


def train_until_convergence(trainer, dataset, max_epochs=None, max_error=0,
                            continue_epochs=10, validation_proportion=0.3):
    epochs = 0

    training_data, validation_data = split_dataset(
        dataset(1 - validation_proportion))
    if not (len(training_data) > 0 and len(validation_data)):
        raise ValueError("Provided dataset too small to be split into training " +
                         "and validation sets with proportion " + str(validation_proportion))

    best_weights = trainer.module.params.copy()
    best_err = trainer.testOnData(validation_data)
    training_errors = []
    validation_errors = [best_err]
    while True:
        training_errors.append(trainer.train())
        validation_errors.append(trainer.testOnData(validation_data))
        if not epochs or validation_errors[-1] < best_err:
            # one update is always done
            best_err = validation_errors[-1]
            if best_err < max_error:
                trainer.module.params[:] = best_weights
                break
            best_weights = trainer.module.params.copy()

        if max_epochs is not None and epochs >= max_epochs:
            trainer.module.params[:] = best_weights
            break
        epochs += 1

        if len(validation_errors) >= continue_epochs:
            new = validation_errors[-continue_epochs:]
            if min(new) > best_err:
                trainer.module.params[:] = best_weights
                break
    training_errors.append(trainer.testOnData(training_data))

    return training_errors, validation_errors


def split_dataset(ds, proportion=0.7):

    indices = random.permutation(len(ds))
    separator = int(len(ds) * proportion)

    left_indices = indices[:separator]
    right_indices = indices[separator:]

    training_set = SupervisedDataSet(inp=ds['input'][left_indices],
                                     target=ds['target'][left_indices])
    testing_set = SupervisedDataSet(inp=ds['input'][right_indices],
                                    target=ds['target'][right_indices])
    return training_set, testing_set
