import numpy as np

from activation_function import *

class HiddenLayer(object):

    def __init__(self, n_in: int, n_out: int, W, b, activation: str):
        
        if W == None:
            rng = np.random.default_rng()

            W = np.zeros((n_out, n_in))
            w_ = 1. / float(n_in)

            for j in range(n_out):
                for i in range(n_in):
                    # initialize W with uniform distribution
                    W[j][i] = rng.uniform(-w_, w_, 1)

        if b == None:
            b = np.zeros(n_out)

        self.n_in = n_in
        self.n_out = n_out
        self.W = W
        self.b = b
        
        if activation == "sigmoid" or activation == None:
            self.activation = lambda x : sigmoid(x)
            self.dactivation = lambda x : dsigmoid(x)
        elif activation == "tanh":
            self.activation = lambda x : tanh(x)
            self.dactivation = lambda x : dtanh(x)
        else:
            print("Error: this activation function is not supported")
            # TODO

    def output(self, x):
        
        y = np.zeros(self.n_out)

        for j in range(self.n_out):
            pre_activation = 0.
            for i in range(self.n_in):
                # W: (n_out, n_in), x: (n_in), b: (n_out)
                pre_activation += self.W[j][i] * x[i]
            pre_activation += self.b[j]

            y[j] = self.activation(pre_activation)

        return y

    def forward(self, x):
        return self.output(x)

    def backward(self, X, Z, d_Y, W_prev, minibatch_size: int, learning_rate: float):

        d_Z = np.zeros((minibatch_size, self.n_out)) # backpropagation error

        grad_W = np.zeros((self.n_out, self.n_in))
        grad_b = np.zeros(self.n_out)

        # train with SGD
        # calculate backpropagation error to get gradient of W, b
        for l in range(minibatch_size):
            for j in range(self.n_out):
                for k in range(len(d_Y[0])): # k < (n_out of previous layer)
                    d_Z[l][j] += W_prev[k][j] * d_Y[l][k]
                d_Z[l][j] *= self.dactivation(Z[l][j])

                for i in range(self.n_in):
                    grad_W[j][i] += d_Z[l][j] * X[l][i]

                grad_b += d_Z[l][j]

        # update params
        for j in range(self.n_out):
            for i in range(self.n_in):
                # gradient descent
                self.W[j][i] -= learning_rate * grad_W[j][i] / minibatch_size
            
            # gradient descent
            self.b[j] -= learning_rate * grad_b[j] / minibatch_size

        return d_Z

