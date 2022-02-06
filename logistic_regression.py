import numpy as np

from activation_function import softmax

class LogisticRegression(object):
    
    def __init__(self, nIn: int, nOut: int):
        self.nIn = nIn
        self.nOut = nOut

        self.W = np.zeros((nOut, nIn)) #[nOut][nIn]
        self.b = np.zeros(nOut)

    def train(self, X, T, miniBatchSize: int, learningRate: float):
        grad_W = np.zeros((self.nOut, self.nIn))
        grad_b = np.zeros(self.nOut)

        dY = np.zeros((miniBatchSize, self.nOut))

        # train with SGD

        # calc gradient of W, b
        for k in range(miniBatchSize):

            predicted_Y_ = self.output(X[k])

            for j in range(self.nOut):
                dY[k][j] = predicted_Y_[j] - T[k][j]

                for i in range(self.nIn):
                    grad_W[j][i] += dY[k][j] * X[k][i]

                grad_b[j] += dY[k][j]

        # update params
        for j in range(self.nOut):
            for i in range(self.nIn):
                self.W[j][i] -= learningRate * grad_W[j][i] / miniBatchSize
            self.b[j] -= learningRate * grad_b[j] / miniBatchSize

        return dY

    # calc \sigma(Wx + b)
    # size = W: (nOut, nIn), x: nIn, b: nOut
    def output(self, x):
        assert x.shape[0] == self.nIn, "x shape must be " + str(self.nIn)
        # Wx + b
        preActivation = np.zeros(self.nOut)
        for j in range(self.nOut):
            for i in range(self.nIn):
                wx =  self.W[j][i] * x[i] 
                preActivation[j] += wx
            preActivation[j] += self.b[j]
        
        return softmax(preActivation, self.nOut)


    def predict(self, x):
        y = self.output(x) # y is a probability vector
        t = np.zeros(self.nOut) # t is a label. y is casted to 0 or 1.

        argmax = np.argmax(y)

        for i in range(self.nOut):
            t[i] = 1 if i == argmax else 0
            
        return t

    