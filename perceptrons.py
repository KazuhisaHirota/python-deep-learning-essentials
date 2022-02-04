import numpy as np

from activation_function import step


class Perceptrons(object):
    
    def __init__(self, nIn: int):
        self.nIn = nIn

        self.w = np.zeros(nIn) # weight vector

    def train(self, x, t: int, learningRate: float):
        
        # check if the data is classified correctly
        c = 0.0
        for i in range(self.nIn):
            c += self.w[i] * x[i] * t

        classified = 0
        if c > 0:
            classified = 1
        else: # apply steepest descent method if the data is wrongly classified
            for i in range(self.nIn):
                self.w[i] += learningRate * x[i] * t
        
        return classified

    # calc \sigma(wx)
    # size = W: nIn, x: nIn
    def predict(self, x):
        assert x.shape[0] == self.nIn, "x shape must be " + str(self.nIn)
        # wx
        preActivation = 0.0
        for i in range(self.nIn):
            preActivation +=  self.w[i] * x[i]
        
        return step(preActivation)

    