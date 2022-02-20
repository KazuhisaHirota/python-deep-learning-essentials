import numpy as np

from activation_function import step


class Perceptrons(object):
    
    def __init__(self, n_in: int):
        self.n_in = n_in

        self.w = np.zeros(n_in) # weight vector

    def train(self, x, t: int, learning_rate: float):
        
        # check if the data is classified correctly
        c = 0.0
        for i in range(self.n_in):
            c += self.w[i] * x[i] * t

        classified = 0 # 0: wrong, 1: correct
        if c > 0: # correct
            classified = 1
        else: # apply steepest descent method if the data is wrongly classified
            for i in range(self.n_in):
                self.w[i] += learning_rate * x[i] * t
        
        return classified

    # calc \sigma(wx)
    # size = W: n_in, x: n_in
    def predict(self, x):
        assert x.shape[0] == self.n_in, "x shape must be " + str(self.n_in)
        # wx
        pre_activation = 0.0
        for i in range(self.n_in):
            pre_activation +=  self.w[i] * x[i]
        
        return step(pre_activation)

    