import numpy as np

from activation_function import sigmoid


class DenoisingAutoencoders(object):

    def __init__(self, n_visible: int, n_hidden: int, W, h_bias, v_bias, rng: np.random.RandomState):

        if rng is None:
            rng = np.random.RandomState(1234)

        if W is None:

            W = np.zeros((n_hidden, n_visible))
            w_ = 1. / float(n_visible)

            for j in range(n_hidden):
                for i in range(n_visible):
                    W[j][i] = rng.uniform(low=-w_, high=w_)

        if h_bias is None:
            h_bias = np.zeros(n_hidden)

        if v_bias is None:
            v_bias = np.zeros(n_visible)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = W
        self.h_bias = h_bias
        self.v_bias = v_bias
        self.rng = rng

    def get_corrupted_input(self, x, corruption_level: float):

        corrupted_input = np.zeros(len(x))

        # add masking noise
        for i in range(len(x)):
            rand_ = self.rng.rand() # [0, 1]
            corrupted_input[i] = 0. if rand_ < corruption_level else x[i]

        return corrupted_input

    # encode the input values x
    def get_hidden_values(self, x):

        z = np.zeros(self.n_hidden)
        # z = sigmoid(Wx + h_bias)
        for j in range(self.n_hidden):
            for i in range(self.n_visible):
                z[j] += self.W[j][i] * x[i]
            
            z[j] += self.h_bias[j]
            z[j] = sigmoid(z[j])

        return z

    # decode the hidden values z
    def get_reconstructed_input(self, z):

        y = np.zeros(self.n_visible)
        # y = sigmoid(Wz + v_bias)
        for i in range(self.n_visible):
            for j in range(self.n_hidden):
                y[i] += self.W[j][i] * z[j]

            y[i] += self.v_bias[i]
            y[i] = sigmoid(y[i])

        return y

    def train(self, X, minibatch_size: int, learning_rate: float, corruption_level: float):

        grad_W = np.zeros((self.n_hidden, self.n_visible))
        grad_h_bias = np.zeros(self.n_hidden)
        grad_v_bias = np.zeros(self.n_visible)

        # train with minibatches
        for n in range(minibatch_size):

            # add noise to original inputs
            corrupted_input = self.get_corrupted_input(X[n], corruption_level)
            # encode
            z = self.get_hidden_values(corrupted_input)
            # decode
            y = self.get_reconstructed_input(z)

            # calculate gradients

            # v_bias
            v_ = np.zeros(self.n_visible)
            for i in range(self.n_visible):
                v_[i] = X[n][i] - y[i] # input - reconstructed input
                grad_v_bias[i] += v_[i]

            # h_bias
            h_ = np.zeros(self.n_hidden)
            for j in range(self.n_hidden):
                for i in range(self.n_visible):
                    h_[j] = self.W[j][i] * (X[n][i] - y[i])  # W: (n_hidden, n_visible)

                h_[j] *= z[j] * (1. - z[j])
                grad_h_bias[j] += h_[j]

            # W
            for j in range(self.n_hidden):
                for i in range(self.n_visible):
                    grad_W[j][i] += h_[j] * corrupted_input[i] + v_[i] * z[j]

        # update params
        for j in range(self.n_hidden):
            for i in range(self.n_visible):
                self.W[j][i] += learning_rate * grad_W[j][i] / minibatch_size

            self.h_bias[j] += learning_rate * grad_h_bias[j] / minibatch_size

        for i in range(self.n_visible):
            self.v_bias[i] += learning_rate * grad_v_bias[i] / minibatch_size

    def reconstruct(self, x):
        # encode
        z = self.get_hidden_values(x)
        # decode
        y = self.get_reconstructed_input(z)
        return y