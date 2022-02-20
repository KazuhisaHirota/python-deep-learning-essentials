import numpy as np


def makeDataSet(start, end, mu1, mu2, answer, x, t):
    for i in range(start, end):
        x[i][0] = np.random.randn() + mu1 # input variable 1
        x[i][1] = np.random.randn() + mu2 # input variable 2
        t[i] = answer

# class 1 : [0, 0], [1, 1] -> Negative [0, 1]
# class 2 : [0, 1], [1, 0] -> Positive [1, 0]
def make_xor_dataset():
    
    X = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])
    
    T = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
    ])
    
    return X, T
