from unittest import TestCase
from mock import patch, call

from speedometer.neural_network import NeuralNetwork


@patch('speedometer.neural_network.Thread')
@patch('speedometer.neural_network.SupervisedDataSet')
@patch('speedometer.neural_network.buildNetwork')
class NeuralNetworkTestCase(TestCase):

    @patch('speedometer.neural_network.NetworkReader')
    def test_creation(self, reader_mock, build_mock, ds_mock, thread_mock):

        net_tuple = (1, 2, 3)
        net = NeuralNetwork(net_tuple)

        reader_mock.readFrom.assert_has_no_calls()
        build_mock.assert_called_once_with(*net_tuple)
        ds_mock.assert_called_once_with(inp=2, target=1)
        thread_mock.assert_called_once_with(target=net.train, args=(1,))

        self.assertFalse(net.done)
        self.assertFalse(net.save)
        self.assertTrue(thread_mock.return_value.daemon)

    def test_creation_with_epochs(self, build_mock, ds_mock, thread_mock):

        net_tuple = (1, 2, 3)
        epochs = 5
        net = NeuralNetwork(net_tuple, epochs)

        build_mock.assert_called_once_with(*net_tuple)
        ds_mock.assert_called_once_with(inp=2, target=1)
        thread_mock.assert_called_once_with(target=net.train, args=(epochs,))

        self.assertFalse(net.done)
        self.assertFalse(net.save)
        self.assertTrue(thread_mock.return_value.daemon)

    @patch('speedometer.neural_network.NetworkReader')
    def test_load(self, reader_mock, build_mock, ds_mock, thread_mock):

        net_tuple = (1, 2, 3)
        epochs = 5
        net = NeuralNetwork(net_tuple, epochs, load='foo', save='bar')

        build_mock.assert_has_no_calls()
        reader_mock.readFrom.assert_called_once_with('foo')
        ds_mock.assert_called_once_with(inp=2, target=1)
        thread_mock.assert_called_once_with(target=net.train, args=(epochs,))

        self.assertTrue(net.save)
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

    @patch('speedometer.neural_network.NetworkWriter')
    @patch('speedometer.neural_network.BackpropTrainer')
    def test_train_with_save(self, trainer_mock, writer_mock, *_):

        epochs = 2
        net = NeuralNetwork((1, 2, 3), save='foo')
        self.assertFalse(net.done)
        self.assertTrue(net.save)

        res = net.train(epochs)

        writer_mock.writeToFile.assert_called_once_with(net.network, 'foo')
        trainer_mock.assert_called_once_with(net.network, net.ds)
        trainer_mock.return_value.trainEpochs.assert_called_once_with(epochs)
        self.assertIsNone(res)
        self.assertTrue(net.done)

    def test_is_done_done(self, *_):

        net = NeuralNetwork((1, 2, 3))
        net.done = True

        res = net.is_done()

        self.assertTrue(res)

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
