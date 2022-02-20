import numpy as np
from numpy.random import seed

from perceptrons import Perceptrons
from dataset import make_dataset


def testPerceptrons():
    
    print("set configs")

    seed(1234)

    train_N = 1000 # number of train_ing data
    train_size = int(train_N/2) # TODO rename
    
    test_N = 200 # number of test data
    test_size = int(test_N/2) # TODO rename
    
    n_in = 2 # dim of input data
    # nOut = 1

    epochs = 100
    learning_rate = 1.0 # learn_ing rate can be 1 in perceptrons

    print("initialize tensors")

    # train_ing data
    train_X = np.zeros((train_N, n_in)) # input data for train_ing
    train_T = np.zeros(train_N) # answers (labels) for train_ing

    # test data
    test_X = np.zeros((test_N, n_in)) # input data for test
    test_T = np.zeros(test_N) # answers (labels) for test
    predicted_T = np.zeros(test_N) # outputs predicted by the model

    print("make data set")

    # class1 inputs x11 and x12: x11 ~ N(-2.0, 1.0), x12 ~ N(+2.0, 1.0)
    mu11, mu12 = -2.0, 2.0
    answer1 = 1
    # make train_ing data
    make_dataset(0, train_size - 1, mu11, mu12, answer1, train_X, train_T)
    # make test data
    make_dataset(0, test_size - 1, mu11, mu12, answer1, test_X, test_T)

    # class2 inputs x21 and x22: x21 ~ N(+2.0, 1.0), x22 ~ N(-2.0, 1.0)
    mu21, mu22 = 2.0, -2.0
    answer2 = -1
    # make train_ing data
    make_dataset(train_size - 1, train_N, mu21, mu22, answer2, train_X, train_T)
    # make test data
    make_dataset(test_size - 1, test_N, mu21, mu22, answer2, test_X, test_T)

    # build the model

    # construct
    classifier = Perceptrons(n_in)

    # train
    print("train data")
    epoch = 0 # train_ing epoch counter
    while True:
        print("epoch: ", str(epoch))

        classified_ = 0
        for i in range(train_N):
            classified_ += classifier.train(train_X[i], train_T[i], learning_rate)
        
        if classified_ == train_N: # when all data are classified correctly
            break

        epoch += 1
        if epoch > epochs:
            break

    # test
    print("test")
    for i in range(test_N):
        predicted_T[i] = classifier.predict(test_X[i])

    # evaluate the model
    print("evaluate")
    confusion_matrix = np.zeros((2, 2))
    accuracy = 0.0
    precision = 0.0
    recall = 0.0

    for i in range(test_N):
        if predicted_T[i] > 0: # positive
            if test_T[i] > 0: # TP
                accuracy += 1
                precision += 1
                recall += 1
                confusion_matrix[0][0] += 1
            else: # FP
                confusion_matrix[1][0] += 1
        else: # negative
            if test_T[i] > 0: #  FN
                confusion_matrix[0][1] += 1
            else: # TN
                accuracy += 1
                confusion_matrix[1][1] += 1

    accuracy /= test_N
    precision /= confusion_matrix[0][0] + confusion_matrix[1][0] # TP / (TP + FP)
    recall /= confusion_matrix[0][0] + confusion_matrix[0][1] # TP / (TP + FN)

    print("Perceptrons model evaluation")
    print("Accuracy: ", str(accuracy * 100))
    print("Precision: ", str(precision * 100))
    print("Recall: ", str(recall * 100))
    

if __name__ == "__main__":
    testPerceptrons()