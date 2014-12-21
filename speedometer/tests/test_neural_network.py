from unittest import TestCase
from mock import patch, call

from speedometer.neural_network import NeuralNetwork


@patch('speedometer.neural_network.Thread')
@patch('speedometer.neural_network.SupervisedDataSet')
@patch('speedometer.neural_network.buildNetwork')
class NeuralNetworkTestCase(TestCase):

    def test_creation(self, build_mock, ds_mock, thread_mock):

        net_tuple = (1, 2, 3)
        net = NeuralNetwork(net_tuple)

        build_mock.assert_called_once_with(*net_tuple)
        ds_mock.assert_called_once_with(2, 1)
        thread_mock.assert_called_once_with(target=net.train, args=(1,))

        self.assertFalse(net.done)
        self.assertTrue(thread_mock.return_value.daemon)

    def test_creation_with_epochs(self, build_mock, ds_mock, thread_mock):

        net_tuple = (1, 2, 3)
        epochs = 5
        net = NeuralNetwork(net_tuple, epochs)

        build_mock.assert_called_once_with(*net_tuple)
        ds_mock.assert_called_once_with(2, 1)
        thread_mock.assert_called_once_with(target=net.train, args=(epochs,))

        self.assertFalse(net.done)
        self.assertTrue(thread_mock.return_value.daemon)

    def test_add_sample(self, *_):

        net = NeuralNetwork((1, 2, 3))

        args_1 = (1, 2, 3)
        args_2 = (3, 4, 5)

        self.assertIsNone(net.add_sample(*args_1))
        self.assertIsNone(net.add_sample(*args_2))

        net.ds.addSample.assert_has_calls([call(args_1[0:2], (args_1[2],)),
                                           call(args_2[0:2], (args_2[2],))])

        self.assertFalse(net.done)

    @patch('speedometer.neural_network.BackpropTrainer')
    def test_train(self, trainer_mock, *_):

        epochs = 2
        net = NeuralNetwork((1, 2, 3))
        self.assertFalse(net.done)

        res = net.train(epochs)

        trainer_mock.assert_called_once_with(net.network, net.ds)
        trainer_mock.return_value.trainEpochs.assert_called_once_with(epochs)
        self.assertIsNone(res)
        self.assertTrue(net.done)

    def test_is_done_done(self, *_):

        net = NeuralNetwork((1, 2, 3))
        net.done = True

        res = net.is_done()

        self.assertTrue(res)
        self.assertFalse(net.done)

    def test_is_done_not_done(self, *_):

        net = NeuralNetwork((1, 2, 3))

        res = net.is_done()

        self.assertFalse(res)
        self.assertFalse(net.done)

    def test_result(self, *_):

        net = NeuralNetwork((1, 2, 3))

        res = net.result(1, 2)

        net.network.activate.assert_called_once_with((1, 2))
        self.assertEqual(res, net.network.activate.return_value)
