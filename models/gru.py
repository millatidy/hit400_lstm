import numpy as np

class GRU:

    def __init__(self, x, y):
        # initialize instance variables
        self.x = x
        self.y = y
        # Randomly initialize network weights
        X_z = np.random.uniform(-np.sqrt(1./self.x), np.sqrt(1./self.y), (self.y, self.x))
        X_r = np.random.uniform(-np.sqrt(1./self.x), np.sqrt(1./self.y), (self.y, self.x))
        X_h = np.random.uniform(-np.sqrt(1./self.x), np.sqrt(1./self.y), (self.y, self.x))
        Y_z = np.random.uniform(-np.sqrt(1./self.y), np.sqrt(1./self.y), (self.y, self.y))
        Y_r = np.random.uniform(-np.sqrt(1./self.y), np.sqrt(1./self.y), (self.y, self.y))
        Y_h = np.random.uniform(-np.sqrt(1./self.y), np.sqrt(1./self.y), (self.y, self.y))
        # Stack arrays and swap axes
        self.X = np.dstack((X_z, X_r, X_h)).swapaxes(0,2)
        self.X = self.X.swapaxes(2,1)
        self.Y = np.dstack((Y_z, Y_r, Y_h)).swapaxes(0,2)
        self.Y = self.Y.swapaxes(2,1)

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
            z_t = self.sigmoid(self.X[0][:, input_data[t]] + self.Y[0].dot(hs[t-1]))
            r_t = self.sigmoid(self.X[1][:, input_data[t]] + self.Y[1].dot(hs[t-1]))
            _hs = np.tanh(self.X[2][:, input_data[t]] + self.Y[2].dot(r_t.dot(hs[t-1])))
            hs[t] = (1 - z_t) * hs[t-1] + z_t.dot(_hs)
        return hs

    def sigmoid(self, x):
        return 1 / 1 + -np.exp(x)

    def calculate_total_loss(self, x, y):
        # Total loss
        Loss = 0
        for i in (len(y)):
            o = self.forward_propagation(x[i])
            cost = x[np.arrange(len(y[i])), y[i]]
            # Add all
            Loss += -1 * np.sum(np.log(cost))
        return Loss

    def calculate_loss(self, x, y):
        # divide total losss by number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

    def bptt(self, x, y):
        T = len(y)
        # Perfom forward_propagation
        o = self.forward_propagation(x)
        # Initialize gradient variables as zero valued arrays
        dLdX = np.zeros(self.X.shape)
        dLdY = np.zeros(self.Y.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        # Back propagate for each output
        for t in np.arrange(T)[::-1]:
