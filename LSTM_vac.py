import numpy as np

class LSTM:

    def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = 4,bias_f,
                 bias_i, bias_s, bias_o):

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.bias_f = bias_f
        self.bias_i = bias_i
        self.bias_s = bias_s
        self.bias_o = bias_o
        # random initialize network weights
        # [-1/sqrt(n), 1/sqrt(n)]
        wf = np.random.uniform(-sqrt(1./word_dim), sqrt(1./word_dim), (hidden_dim, word_dim))
        wi = np.random()
        w_s = np.random()
        wo = np.random()

    def forward_propagation(self, input):
        # N is the number of steps
        N = len(input)

        h_state_t_prev = 0.0
        state_prev = 0.0

        for x in input:

            f_gate_t = self.sigmoid(self.wf, self.h_state_t_prev, x, self.bias_f)

            i_gate_t = self.sigmoid(self.wi, self.h_state_t_prev, x, self.bias_i)
            _state_t = self.sigmoid(self.w_s, self.h_state_t_prev, x, self.bias_s)
            state_t = (f_gate_t * state_prev) + (i_gate_t * _state_t)

            out_t = self.sigmoid(self.wo, self.h_state_t_prev, x, self.bias_o)
            h_state_t = tanh(state_t) * self.out_t

        return out_t, h_state_t


    def sigmoid(self, w, p_s, i, b):
        cal = (w * p_s) + (w * i) + b
        sig = 1 / (1 + -exp(cal))

        return sig

    # This block of code is copieed from WILDML tutorial and must be understood
    # personalised at a later stage of the project
    # [BLOCK start]
    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)

        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N
    # [BLOCK end]


    def bptt(self, x, y): 
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
