import theano
import theano.tensor as T
import numpy as np

floatX = theano.config.floatX

class TheanoRNN:

    def __init__(self, io_dim, hidden_dim):
        self.io_dim = io_dim
        self.hidden_dim = hidden_dim

        U = np.random.uniform(-np.sqrt(1. / io_dim), np.sqrt(1. / io_dim), (hidden_dim, io_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (io_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

        self.U = theano.shared(name='U', value=U.astype(np.float32))
        self.V = theano.shared(name='V', value=V.astype(np.float32))
        self.W = theano.shared(name='W', value=W.astype(np.float32))
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')

        def forward_propagation_step(x_t, h_prev, U, V, W):
            h_t = T.tanh(U[:,x_t] + W.dot(h_prev))
            o = T.nnet.softmax(V.dot(h))
            return [o[0], h_t]

        [o, h]; _ = theano.scan(
            forward_propagation_step,
            sequences=[x],
            non_sequences=[U, V, W],
            outputs_info=[None, dict(initialize=T.zeros(self.hidden_dim))],
            strict=True
        )

        prediction = T.argmax(o, axis=1)

        self.predict = theano.function([x], prediction)
