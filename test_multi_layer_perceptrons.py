import numpy as np

from multi_layer_perceptrons import MultiLayerPerceptrons
from dataset import make_xor_dataset


def test_multi_layer_perceptrons():

    print("set configs")

    np.random.seed(123)

    patterns = 2
    train_N = 4
    test_N = 4
    n_in = 2
    n_hidden = 3
    n_out = patterns

    epochs = 5000
    learning_rate = 0.1

    minibatch_size = 1 # here, we do on-line training
    minibatch_N = int(train_N / minibatch_size)

    # make XOR dataset
    print("make dataset")
    train_X, train_T = make_xor_dataset()
    test_X, test_T = make_xor_dataset()

    # make minibatches
    print("make minibatches")

    minibatch_index = []
    for i in range(train_N):
        minibatch_index.append(i)
    np.random.shuffle(minibatch_index) # shuffle data index for SGD

    minibatch_train_X = np.zeros((minibatch_N, minibatch_size, n_in))
    minibatch_train_T = np.zeros((minibatch_N, minibatch_size, n_out))
    for i in range(minibatch_N):
        for j in range(minibatch_size):
            position = i * minibatch_size + j
            minibatch_train_X[i][j] = train_X[minibatch_index[position]]
            minibatch_train_T[i][j] = train_T[minibatch_index[position]]

    # build Multi-Layer Perceptrons model

    # construct
    classifier = MultiLayerPerceptrons(n_in, n_hidden, n_out)

    # train
    print("train")
    for epoch in range(epochs):
        print("epoch: " + str(epoch))
        for batch in range(minibatch_N):
            classifier.train(minibatch_train_X[batch],
                             minibatch_train_T[batch],
                             minibatch_size, learning_rate)

    # test
    print("test")
    predicted_T = np.zeros((test_N, n_out))
    for i in range(test_N):
        predicted_T[i] = classifier.predict(test_X[i])

    # evaluate the model
    print("evaluate")
    confusion_matrix = np.zeros((patterns, patterns))
    accuracy = 0.0
    precision = np.zeros(patterns)
    recall = np.zeros(patterns)

    for i in range(test_N):
        predicted_ = np.where(predicted_T[i] == 1) # find the position of the value 1
        actual_ = np.where(test_T[i] == 1) # find the position of the value 1
        col = predicted_[0][0] # NOTE
        row = actual_[0][0] # NOTE
        confusion_matrix[row][col] += 1

    for i in range(patterns):
        col_ = 0.0
        row_ = 0.0

        for j in range(patterns):
            if i == j:
                accuracy += confusion_matrix[i][j]
                precision[i] += confusion_matrix[j][i] # NOTE
                recall[i] += confusion_matrix[i][j]

            col_ += confusion_matrix[j][i] # NOTE
            row_ += confusion_matrix[i][j]

        precision[i] /= col_
        recall[i] /= row_

    accuracy /= test_N

    print("MLP model evaluation")
    print("Accuracy: " + str(accuracy * 100))
    print("Precision: ")
    for i in range(patterns):
        print("class: " + str(i+1) + ", precision: " + str(precision[i] * 100))
    print("Recall: ")
    for i in range(patterns):
        print("class: " + str(i+1) + ", recall: " + str(recall[i] * 100))
    

if __name__ == "__main__":
    test_multi_layer_perceptrons()
    