import numpy as np
import copy

from hidden_layer import HiddenLayer
from restricted_boltzmann_machines import RestrictedBoltzmannMachines
from logistic_regression import LogisticRegression


class DeepBeliefNets(object):

    def __init__(self, n_in: int, hidden_layer_sizes, n_out: int, rng: np.random.RandomState):

        if rng is None:
            rng = np.random.RandomState(1234)

        self.n_in = n_in
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_out = n_out
        self.n_layers = len(hidden_layer_sizes)
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.rng = rng

        # construct multi-layer
        for i in range(self.n_layers):
            hidden_layer_n_in = n_in if i == 0 else hidden_layer_sizes[i-1]

            # construct hidden layers with sigmoid function
            # weight matrices and bias vectors will be shared with RBM layers
            self.sigmoid_layers.append(
                HiddenLayer(hidden_layer_n_in, hidden_layer_sizes[i],
                            W=None, b=None, rng=rng, activation="sigmoid"))

            # construct RBM layers
            self.rbm_layers.append(
                RestrictedBoltzmannMachines(hidden_layer_n_in, hidden_layer_sizes[i],
                                            W=self.sigmoid_layers[i].W,
                                            h_bias=self.sigmoid_layers[i].b,
                                            v_bias=None, rng=rng))

        # logistic regression layer for output
        self.logistic_layer = LogisticRegression(n_in=hidden_layer_sizes[self.n_layers-1], # the last hidden layer
                                                 n_out=n_out)

    # not use labels T
    def pretrain(self, X, minibatch_size: int, minibatch_N: int, epochs: int, learning_rate: float, k: int):

        for layer in range(self.n_layers): # pre-train layer-wise
            print("layer=", layer)
            for epoch in range(epochs):
                print("epoch=", epoch)
                for batch in range(minibatch_N):
                    
                    X_ = np.zeros((minibatch_size, self.n_in))

                    # set input data for current layer
                    if layer == 0:
                        X_ = X[batch]
                    else:
                        prev_layer_X = X_
                        X_ = np.zeros((minibatch_size, self.hidden_layer_sizes[layer-1]))

                        for i in range(minibatch_size):
                            X_[i] = self.sigmoid_layers[layer-1].output_binomial(prev_layer_X[i], self.rng)

                    self.rbm_layers[layer].contrastive_divergence(X_, minibatch_size, learning_rate, k)

    # use labels T
    def finetune(self, X, T, minibatch_size: int, learning_rate: float):

        layer_inputs = []
        layer_inputs.append(X)

        Z = None
        
        # forward hidden layers
        for layer in range(self.n_layers):

            x_ = [] # layer input
            Z_ = np.zeros((minibatch_size, self.hidden_layer_sizes[layer]))

            for n in range(minibatch_size):

                x_ = X[n] if layer == 0 else Z[n]
                Z_[n] = self.sigmoid_layers[layer].forward(x_)

            Z = Z_
            layer_inputs.append(copy.deepcopy(Z)) # layerInputs.add(Z.clone())

        # forward & backward output layer
        d_Y = self.logistic_layer.train(Z, T, minibatch_size, learning_rate)

        # backward hidden layers
        d_Z = None
        for layer in reversed(range(self.n_layers)):

            W_prev = None
            if layer == self.n_layers - 1:
                W_prev = self.logistic_layer.W # output layer
            else:
                W_prev = self.sigmoid_layers[layer + 1].W # hidden layers
                d_Y = copy.deepcopy(d_Z) # dY = dZ.clone()

            d_Z = self.sigmoid_layers[layer].backward(
                layer_inputs[layer], # X
                layer_inputs[layer + 1], # Z
                d_Y, W_prev, minibatch_size, learning_rate)

    def predict(self, x):

        z = None

        for layer in range(self.n_layers):

            x_ = x if layer == 0 else copy.deepcopy(z) # x_ = z.clone()

            z = self.sigmoid_layers[layer].forward(x_)

        return self.logistic_layer.predict(z)