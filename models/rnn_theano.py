import numpy as np

class RNN_THEANO:
    '''
        input_dim is the array size of the input data
        hidden_dim is the array size of the hidden input_dim
        output_dim is the array size of the output
        # input weights is array [input_dim,hidden_dim]
        # hidden weights is array [hidden_dim, hidden_dim]
        # output weights is array [hidden_dim, output_dim]
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        # assign instance variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # randomly initialize network weights
        self.input_to_hidden_weights = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        self.hidden_to_hidden_weights = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.hidden_to_output_weights = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))


    def forward_propagation(self, input_array):
        # N is the number of steps in our forward_propagation
        # N is to be looked at and modified later on
        N = len(input_array)
        # we save all our hideen states since we need to use them later
        # We add one additional element for the initial hidden state,
        # which we set to zero
        hidden_state = np.zeros((N + 1, self.hidden_dim))
        hidden_state[-1] = np.zeros(self.hidden_dim)
        # We save the output of each step for later use
        output = np.zeros((N, self.input_dim))
        # For each step
        for t in np.arange(N):
            # bias to be added later
            hidden_state[t] = np.tanh(self.input_to_hidden_weights[:,input_array[t]]  + self.hidden_to_hidden_weights.dot(hidden_state[-1]))
            output[t] = self.softmax(self.hidden_to_output_weights.dot(hidden_state[t]))

        return (output, hidden_state)

    def softmax(self, z):
        # to be implimented
        return z
