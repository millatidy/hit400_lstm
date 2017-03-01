import numpy as np

class GRU:

    def __init__(self, x, y):
        # initialize instance variables
        self.x = x
        self.y = y
        # Randomly initialize network weights
        self.X_z = np.random.uniform(-np.sqrt(1./self.x), np.sqrt(1./self.y), (self.y, self.x))
        self.X_r = np.random.uniform(-np.sqrt(1./self.x), np.sqrt(1./self.y), (self.y, self.x))
        self.X_h = np.random.uniform(-np.sqrt(1./self.x), np.sqrt(1./self.y), (self.y, self.x))
        self.Y_z = np.random.uniform(-np.sqrt(1./self.y), np.sqrt(1./self.y), (self.y, self.y))
        self.Y_r = np.random.uniform(-np.sqrt(1./self.y), np.sqrt(1./self.y), (self.y, self.y))
        self.Y_h = np.random.uniform(-np.sqrt(1./self.y), np.sqrt(1./self.y), (self.y, self.y))

    def forward_propagation(self, input_data):
        # steps in forward propagation
        N = len(input_data)
        # we save all our hideen states since we need to use them later
        # We add one additional element for the initial hidden state,
        # which we set to zero
        hs = np.zeros((N + 1, self.y))
        hs[-1] = np.zeros(self.y)
        # We save the output of each step for later use
        o = np.zeros((N, self.x))
        # For each step
        for t in np.arange(N):
            z_t = self.sigmoid(self.X_z[:, input_data[t]] + self.Y_z.dot(hs[t-1]))
            r_t = self.sigmoid(self.X_r[:, input_data[t]] + self.Y_r.dot(hs[t-1]))
            _hs = np.tanh(self.X_h[:, input_data[t]] + self.Y_h.dot(r_t.dot(hs[t-1])))
            hs[t] = (1 - z_t) * hs[t-1] + z_t.dot(_hs)
        return hs

    def sigmoid(self, x):
        return 1 / 1 + -np.exp(x)

    # THis code is taken from WILDML website and is pasted here for
    # progress purposes.
    # Will get back to it
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
