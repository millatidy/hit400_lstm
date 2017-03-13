import unittest
import numpy as np
import time as time

from models.rnn import RNN
from models.rnn_theano import RNN_THEANO
from models.gru import GRU

class TestCase(unittest.TestCase):
    pass

    # Vanila RNN TestCASE
    def test_vanilla_rnn(self):
        rnn = RNN(8000,100, 8000)
        input_data = np.arange(8000)
        t0 = time.time()
        assert rnn.forward_propagation(input_data)
        tt = time.time() - t0
        print("\nRNN forward propagation %s sec\n" %str(tt))


    # # Vanila Theano RNN TestCASE
    # def test_initialization_v_theano_rnn(self):
    #     rnn_t = RNN_THEANO(8000,100, 8000)
    #     assert rnn_t.input_dim == 8000
    #     assert rnn_t.hidden_dim == 100
    #     assert rnn_t.output_dim == 8000
    #     assert rnn_t.IH.dtype == "float64"
    #     assert rnn_t.HH.dtype == "float64"
    #     assert rnn_t.HO.dtype == "float64"

    # def test_forward_propagation_v_theano_rnn(self):
    #     rnn_t = RNN_THEANO(8000,100, 8000)
    #     input_data = np.arange(8000)
    #     assert rnn_t.forward_propagation(input_data)

    def test_vanilla_gru(self):
        gru = GRU(8000,100)
        assert gru.X[0].dtype == "float64"
        assert gru.X[1].dtype == "float64"
        assert gru.X[2].dtype == "float64"
        assert gru.Y[0].dtype == "float64"
        assert gru.Y[1].dtype == "float64"
        assert gru.Y[2].dtype == "float64"
        input_data = np.arange(8000)
        t0 = time.time()
        gru.forward_propagation(input_data)
        tt = time.time() - t0
        print("\nGRU forward propagation %s sec\n" %str(tt))
        # print("Expected loss from random predictions %d" %np.log(8000))
        # print("Actual loss %d" %gru.calculate_loss(np.arange(1000).reshape(2,2), np.arange(1000).reshape(2,2)))

    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()
