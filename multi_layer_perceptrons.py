import numpy as np

from hidden_layer import HiddenLayer
from logistic_regression import LogisticRegression


class MultiLayerPerceptrons(object):

    def __init__(self, n_in: int, n_hidden: int, n_out: int):

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        # construct hidden layer with tanh as activation function
        activation_func = "tanh" # sigmoid or tanh
        self.hidden_layer = HiddenLayer(n_in, n_hidden, W=None, b=None, rng=None, activation=activation_func)

        # construct output layer i.e. multi-class logistic layer
        self.logistic_layer = LogisticRegression(n_hidden, n_out)

    def train(self, X, T, minibatch_size: int, learning_rate: float):
        # outputs of the hidden layer (= inputs of the output layer)
        Z = np.zeros((minibatch_size, self.n_hidden)) # NOTE "self.n_in" in the original code is wrong
        
        # forward the hidden layer: X => Z
        for n in range(minibatch_size):
            # Z[n]: (n_hidden)
            # n_hidden is the size of the inputs of the output layer
            Z[n] = self.hidden_layer.forward(X[n]) # activate input units

        # forward & backward the output layer
        d_Y = self.logistic_layer.train(Z, T, minibatch_size, learning_rate)

        prev_W = self.logistic_layer.W
        # backward the hidden layer (backpropagation)
        self.hidden_layer.backward(X, Z, d_Y, prev_W, minibatch_size, learning_rate)

    def predict(self, x):

        z = self.hidden_layer.output(x)
        return self.logistic_layer.predict(z)