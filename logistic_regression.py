import numpy as np

from activation_function import softmax


class LogisticRegression(object):
    
    def __init__(self, n_in: int, n_out: int):
        self.n_in = n_in
        self.n_out = n_out

        self.W = np.zeros((n_out, n_in)) #[n_out][n_in]
        self.b = np.zeros(n_out)

    def train(self, X, T, minibatch_size: int, learning_rate: float):
        grad_W = np.zeros((self.n_out, self.n_in))
        grad_b = np.zeros(self.n_out)

        d_Y = np.zeros((minibatch_size, self.n_out))

        # train with SGD

        # calc gradient of W, b
        for k in range(minibatch_size):

            predicted_Y_ = self.output(X[k])

            for j in range(self.n_out):
                d_Y[k][j] = predicted_Y_[j] - T[k][j]

                for i in range(self.n_in):
                    grad_W[j][i] += d_Y[k][j] * X[k][i]

                grad_b[j] += d_Y[k][j]

        # update params
        for j in range(self.n_out):
            for i in range(self.n_in):
                self.W[j][i] -= learning_rate * grad_W[j][i] / minibatch_size
            self.b[j] -= learning_rate * grad_b[j] / minibatch_size

        return d_Y

    # calc \sigma(Wx + b)
    # size = W: (n_out, n_in), x: n_in, b: n_out
    def output(self, x):
        assert x.shape[0] == self.n_in, "x shape must be " + str(self.n_in)
        # Wx + b
        pre_activation = np.zeros(self.n_out)
        for j in range(self.n_out):
            for i in range(self.n_in):
                wx =  self.W[j][i] * x[i] 
                pre_activation[j] += wx
            pre_activation[j] += self.b[j]
        
        return softmax(pre_activation, self.n_out)


    def predict(self, x):
        y = self.output(x) # y is a probability vector
        t = np.zeros(self.n_out) # t is a label. y is casted to 0 or 1.

        argmax = np.argmax(y)

        for i in range(self.n_out):
            t[i] = 1 if i == argmax else 0
            
        return t

    