from numpy.random import randn 


def makeDataSet(start, end, mu1, mu2, answer, x, t):
    for i in range(start, end):
        x[i][0] = randn() + mu1 # input variable 1
        x[i][1] = randn() + mu2 # input variable 2
        t[i] = answer