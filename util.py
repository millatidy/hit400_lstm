import sys
import operator
import numpy as np
from models.TheanoGRU import TheanoGRU

from gensim.models import word2vec
vectors = word2vec.Word2Vec.load("dataset/300features_40minwords_10context_movie_lines")

def load_data(filename):
    # Data must be read in pairs i.e. utterrance response pair
    input_x = []
    target_y = []
    print("Reading Training data...")
    training_data = open(filename, 'r')
    for line in training_data:
        # split the sentance into utterrance and response
        # at the "<end>" token
        utterrance, response = line.split('<end>')
        input_x.append(utterrance)
        target_y.append(response)
    # convert words to vectrs
    input_x = np.asarray([word_to_vec(input_x)])
    target_y = np.asarray([word_to_vec(target_y)])

    return (input_x, target_y)

def word_to_vec(list_of_sentences):
    # list comprehesion convert words to vectors
    # [[word = vectors[word] for word int sentence] for sentence in list_of_sentences]
    list_of_vectors = []
    for word in list_of_sentences:
        pass
        # vector = vectors[word]
        # list_of_vectors.append(vector)
    # return list_of_vectors
    return vectors['hi'].astype('int32')

def vec_to_word():
    pass

def train_model(model, input_x, target_y, learning_rate=0.001, nepoch=20, decay=0.9, callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(target_y)):
            # One SGD step
            model.sgd_step(input_x[i], target_y[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen)
    return model

def save_parameters(model, outfile):
    np.savez(outfile,
        E=model.E.get_value(),
        U=model.U.get_value(),
        W=model.W.get_value())

    print"Saved model parameters to %s." % outfile

def load_parameters(path, modelClass=TheanoGRU):
    npzfile = np.load(path)
    E, U, W, V = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    return model

def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['E', 'U', 'W', 'b', 'V', 'c']
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

def generate_response():
    pass
