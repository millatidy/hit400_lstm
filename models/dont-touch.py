ort theano
import theano.tensor as T
import numpy as np

floatX = theano.config.floatX # pick system architecture at runtime
# floatX = np.float32 # when running on GPU
# floatX = np.float64 # when running on CPU

from gensim.models import word2vec
vectors = word2vec.Word2Vec.load("dataset/300features_40minwords_10context_movie_lines")
# Ask this question on stackoverflow

class TheanoGRU:
    def __init__(self, io_dim, hidden_dim=128, bppt_truncate=-1):
        # assign class variables
        self.io_dim = io_dim
        self.hidden_dim = hidden_dim
        self.bppt_truncate = bppt_truncate
        # initialize weights
        E = np.random.uniform(-np.sqrt(1./self.io_dim), np.sqrt(1./self.io_dim), (self.hidden_dim, ))
        W = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (3, self.hidden_dim, self.hidden_dim))
        U = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (3, self.hidden_dim, self.hidden_dim))
        V = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))

        # Create theano variables
        self.E = theano.shared(name='E', value=E.astype(floatX))
        self.W = theano.shared(name='W', value=W.astype(floatX))
        self.U = theano.shared(name='U', value=U.astype(floatX))
        self.V = theano.shared(name='V', value=V.astype(floatX))

        # SGD
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(floatX))

        # Store our graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, W, U, V = self.E, self.W, self.U, self.V

        x = T.fvector('x')
        y = T.fvector('y')

        # initial hidden vector
        initial_hidden_vector = np.zeros(self.hidden_dim)

        def calculate(x, h_t_prev, E, W, U, V):
            x_t = T.dot(E,x)

            P = U[0].dot(h_t_prev)

            z_t = T.nnet.sigmoid(T.dot(W[0], x_t) +  U[0].dot(h_t_prev))
            r_t = T.nnet.sigmoid(T.dot(W[1], x_t) + U[1].dot(h_t_prev))
            _h_t = T.tanh(T.dot(W[2], x_t) + U[2].dot(h_t_prev * r_t))
            h_t = (T.ones_like(z_t) - z_t) * h_t_prev + z_t * _h_t

            # softmax returns a matrix thith one row only
            # the row we want
            o_t = T.nnet.softmax(V.dot(h_t))[0][0]

            return [o_t, h_t_prev]

        [o, h] , updates = theano.scan(
            calculate,
            # outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            outputs_info=[None, initial_hidden_vector],
            non_sequences = [E, W, U, V],
            sequences=x,
        )

        prediction = T.argmax(o, axis=0)
        prediction_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # Total cost (Regularization can be done here)
        cost = prediction_error

        # gradients
        dE = T.grad(cost, E)
        dW = T.grad(cost, W)
        dU = T.grad(cost, U)
        dV = T.grad(cost, V)

        # assign functions
        self.predict = theano.function([x], o)
        self.prediction_class = theano.function([x], prediction)
        self.c_error = theano.function([x,y], cost)
        self.bptt = theano.function([x, y], [dW, dU, dV])

        # SDG parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2

        self.sgd_step = theano.function(
                [x, y, learning_rate, theano.In(decay, value=0.9)],
                [],
                updates = [
                            (E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                            (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                            (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                            (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                            # (self.mE, mE),
                            (self.mU, mU),
                            (self.mW, mW),
                            (self.mV, mV)
                ]
        )

    def calculate_total_loss(self, X, Y):
        return np.sum([self.c_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self, X, Y):
        # Divide total loss by number of targets
        total_target = np.sum(len(y) for y in Y)
        return self.calculate_total_loss(X,Y)/float(total_target)
