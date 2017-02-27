import numpy as np
import theano as theano
import theano.tensor as T
from utils import *
import operator

class RNN_THEANO:
    '''
        input_dim is the array size of the input data
        hidden_dim is the array size of the hidden input_dim
        output_dim is the array size of the output
        # input weights is array [input_dim,hidden_dim]
        # hidden weights is array [hidden_dim, hidden_dim]
        # output weights is array [hidden_dim, output_dim]
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, bptt_truncate=4):
        # assign instance variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bptt_truncate = bptt_truncate
        # randomly initialize network weights as
        input_to_hidden_weights = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        hidden_to_hidden_weights = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        hidden_to_output_weights = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
        # Theano: Create share variables
        self.IH = theano.shared(name='IH', value=input_to_hidden_weights.astype(theano.config.floatX))
        self.HH = theano.shared(name='HH', value=hidden_to_hidden_weights.astype(theano.config.floatX))
        self.HO = theano.shared(name='HO', value=hidden_to_output_weights.astype(theano.config.floatX))
        # We store theano graph = {}
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        IH, HH, HO = self.IH, self.HH, self.HO
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, h_t_prev, IH, HH, HO):
            h_t = T.tanh(IH[:,x_t] + HH.dot(h_t_prev))
            o_t = T.nnet.softmax(HO.dot(h_t))
            return [o_t[0], h_t]
        [o,h], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[IH, HH, HO],
            truncate_gradient=self.bptt_truncate,
            strict=True)

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o,y))

        # Gradients
        dIH = T.grad(o_error, IH)
        dHH = T.grad(o_error, HH)
        dHO = T.grad(o_error, HO)

        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x,y], o_error)
        self.bptt = theano.function([x,y], [dIH, dHH, HO])

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sdg_step = theano.function([x,y,learning_rate], [],
                        updates=[(self.IH, self.IH - learning_rate * dIH),
                                (self.HH, self.HH - learning_rate * dHH),
                                (self.HO, self.HH - learning_rate * dHO)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)


def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)
