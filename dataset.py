import numpy as np


def make_dataset(start: int, end: int, mu1: float, mu2: float, answer: int, x, t):
    
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

'''
Create training data and test data for demo.
Data without noise would be:
    class 1 : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    class 2 : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    class 3 : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
and to each data, we add some noise.
For example, one of the data in class 1 could be:
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
'''
def make_binomial_dataset(X, patterns: int, train_N_each: int,
                          n_visible_each: int, p_noise: float, rng):
    
    n_visible = n_visible_each * patterns
    
    for pattern in range(patterns):
        for n in range(train_N_each):
            n_ = pattern * train_N_each + n

            for i in range(n_visible):
                if train_N_each * pattern <= n_ and n_ < train_N_each * (pattern + 1) \
                    and n_visible_each * pattern <= i and i < n_visible_each * (pattern + 1):

                    X[n_][i] = rng.binomial(n=1, p=1.-p_noise)

                else:
                    X[n_][i] = rng.binomial(n=1, p=p_noise)
