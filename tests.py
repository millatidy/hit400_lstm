import unittest
import numpy as np

from models.rnn import RNN
from models.rnn_theano import RNN_THEANO

class TestCase(unittest.TestCase):
    def setUp(self):
        rnn = RNN(8000,100, 8000)
        input_data = np.arange(15000)

    # # Vanila RNN TestCASE
    # def test_initialization_v_rnn(self):
    #     rnn = RNN(8000,100, 8000)
    #     assert rnn.input_dim == 8000
    #     assert rnn.hidden_dim == 100
    #     assert rnn.output_dim == 8000
    #     assert rnn.input_to_hidden_weights.dtype == "float64"
    #     assert rnn.hidden_to_hidden_weights.dtype == "float64"
    #     assert rnn.hidden_to_output_weights.dtype == "float64"
    #
    # def test_forward_propagation_v_rnn(self):
    #     rnn = RNN(8000,100, 8000)
    #     input_data = np.arange(8000)
    #     assert rnn.forward_propagation(input_data)


    # Vanila Theano RNN TestCASE
    def test_initialization_v_theano_rnn(self):
        rnn_t = RNN_THEANO(8000,100, 8000)
        assert rnn_t.input_dim == 8000
        assert rnn_t.hidden_dim == 100
        assert rnn_t.output_dim == 8000
        assert rnn_t.input_to_hidden_weights.dtype == "float64"
        assert rnn_t.hidden_to_hidden_weights.dtype == "float64"
        assert rnn_t.hidden_to_output_weights.dtype == "float64"

    def test_forward_propagation_v_theano_rnn(self):
        rnn_t = RNN_THEANO(8000,100, 8000)
        input_data = np.arange(8000)
        assert rnn_t.forward_propagation(input_data)

    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()
