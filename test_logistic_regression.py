import numpy as np
from numpy.random import seed, shuffle

from logistic_regression import LogisticRegression
from data_set import makeDataSet


def testLogisticRegression():
    
    print("set configs")

    seed(1234)

    patterns = 3 # number of classes

    train_size = 400 # TODO rename
    train_N = train_size * patterns
    
    test_size = 60 # TODO rename
    test_N = test_size * patterns
    
    nIn = 2
    nOut = patterns

    print("initialize tensors")

    train_X = np.zeros((train_N, nIn)) # inputs
    train_T = np.zeros((train_N, nOut)) # answers

    test_X = np.zeros((test_N, nIn))
    test_T = np.zeros((test_N, nOut))
    predicted_T = np.zeros((test_N, nOut))

    epochs = 100
    learningRate = 0.2

    minibatchSize = 50 # number of data in each minibatch
    minibatch_N = int(train_N / minibatchSize) # number of minibatches

    train_X_minibatch = np.zeros((minibatch_N, minibatchSize, nIn))
    train_T_minibatch = np.zeros((minibatch_N, minibatchSize, nOut))
    
    minibatchIndex = []
    for i in range(train_N):
        minibatchIndex.append(i)
    shuffle(minibatchIndex) # shuffle data index for SGD

    print("make data set")

    # class1 inputs x11 and x12: x11 ~ N(-2.0, 1.0), x12 ~ N(+2.0, 1.0)
    mu11, mu12 = -2.0, 2.0
    answer1 = np.array([1, 0, 0])
    # make training data
    makeDataSet(0, train_size - 1, mu11, mu12, answer1, train_X, train_T)
    # make test data
    makeDataSet(0, test_size - 1, mu11, mu12, answer1, test_X, test_T)

    # class2 inputs x21 and x22: x21 ~ N(+2.0, 1.0), x22 ~ N(-2.0, 1.0)
    mu21, mu22 = 2.0, -2.0
    answer2 = np.array([0, 1, 0])
    # make training data
    makeDataSet(train_size - 1, train_size * 2 - 1, mu21, mu22, answer2, train_X, train_T)
    # make test data
    makeDataSet(test_size - 1, test_size * 2 - 1, mu21, mu22, answer2, test_X, test_T)

    # class3 inputs x31 and x32: x31 ~ N(0.0, 1.0), x32 ~ N(0.0, 1.0)
    mu31, mu32 = 0.0, 0.0
    answer3 = np.array([0, 0, 1])
    # make training data
    makeDataSet(train_size * 2 - 1, train_N, mu31, mu32, answer3, train_X, train_T)
    # make test data
    makeDataSet(test_size * 2 - 1, test_N, mu31, mu32, answer3, test_X, test_T)

    print("make minibatches")

    # create minibatches with training data
    for i in range(minibatch_N):
        for j in range(minibatchSize):
            train_X_minibatch[i][j] = train_X[minibatchIndex[i * minibatchSize + j]]
            train_T_minibatch[i][j] = train_T[minibatchIndex[i * minibatchSize + j]]

    # build the model

    # construct
    classifier = LogisticRegression(nIn, nOut)

    # train
    print("train data")
    for epoch in range(epochs):
        print("epoch: " + str(epoch))
        for batch in range(minibatch_N):
            classifier.train(train_X_minibatch[batch],
                             train_T_minibatch[batch],
                             minibatchSize, learningRate)
        learningRate *= 0.95 # learning rate is decayed

    # test
    print("test")
    for i in range(test_N):
        predicted_T[i] = classifier.predict(test_X[i])

    # evaluate the model
    print("evaluate")
    confusionMatrix = np.zeros((patterns, patterns))
    accuracy = 0.0
    precision = np.zeros(patterns)
    recall = np.zeros(patterns)

    for i in range(test_N):
        predicted_ = np.where(predicted_T[i] == 1) # find the position of the value 1
        actual_ = np.where(test_T[i] == 1) # find the position of the value 1
        col = predicted_[0][0] # NOTE
        row = actual_[0][0] # NOTE
        confusionMatrix[row][col] += 1

    for i in range(patterns):
        col_ = 0.0
        row_ = 0.0

        for j in range(patterns):
            if i == j:
                accuracy += confusionMatrix[i][j]
                precision[i] += confusionMatrix[j][i] # NOTE
                recall[i] += confusionMatrix[i][j]

            col_ += confusionMatrix[j][i] # NOTE
            row_ += confusionMatrix[i][j]

        precision[i] /= col_
        recall[i] /= row_

    accuracy /= test_N

    print("Logistic Regression model evaluation")
    print("Accuracy: " + str(accuracy * 100))
    print("Precision: ")
    for i in range(patterns):
        print("class: " + str(i+1) + ", precision: " + str(precision[i] * 100))
    print("Recall: ")
    for i in range(patterns):
        print("class: " + str(i+1) + ", recall: " + str(recall[i] * 100))
    

if __name__ == "__main__":
    testLogisticRegression()