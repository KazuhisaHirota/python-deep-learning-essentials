import numpy as np

from stacked_denoising_autoencoders import StackedDenoisingAutoencoders
from dataset import make_binomial_dataset, make_binomial_labels

def test_stacked_denoising_autoencoders():

    print("set configs")

    rng = np.random.RandomState(123)

    train_N_each = 20 # 200: slow
    validation_N_each = 20 # 200: slow
    test_N_each = 150 # 50

    n_in_each = 20

    p_noise_training = 0.2
    p_noise_test = 0.25

    patterns = 3

    train_N = train_N_each * patterns
    validation_N = validation_N_each * patterns
    test_N = test_N_each * patterns

    n_in = n_in_each * patterns
    n_out = patterns
    hidden_layer_sizes = [20, 20]
    corruption_level = 0.3

    pretrain_epochs = 10 # 1000: slow
    pretrain_learning_rate = 0.2
    finetune_epochs = 10 # 1000: slow
    finetune_learning_rate = 0.15

    minibatch_size = 5 # 50: slow
    train_minibatch_N = int(train_N / minibatch_size)
    validation_minibatch_N = int(validation_N / minibatch_size)

    print("make dataset")
    
    train_X = np.zeros((train_N, n_in))
    validation_X = np.zeros((validation_N, n_in))
    test_X = np.zeros((test_N, n_in))
    validation_T = np.zeros((validation_N, n_out))
    test_T = np.zeros((test_N, n_out))
    
    make_binomial_dataset(train_X, patterns, train_N_each, n_in_each, p_noise_training, rng)
    make_binomial_dataset(validation_X, patterns, validation_N_each, n_in_each, p_noise_training, rng)
    make_binomial_dataset(test_X, patterns, test_N_each, n_in_each, p_noise_test, rng)
    
    make_binomial_labels(validation_T, patterns, validation_N_each, n_out)
    make_binomial_labels(test_T, patterns, test_N_each, n_out)

    print("make minibatches")
    
    minibatch_index = [i for i in range(train_N)]
    rng.shuffle(minibatch_index)
    
    train_X_minibatch = np.zeros((train_minibatch_N, minibatch_size, n_in))
    validation_X_minibatch = np.zeros((validation_minibatch_N, minibatch_size, n_in)) 
    validation_T_minibatch = np.zeros((validation_minibatch_N, minibatch_size, n_out))
    
    for j in range(minibatch_size):
        for i in range(train_minibatch_N):
            train_X_minibatch[i][j] = train_X[minibatch_index[i*minibatch_size + j]]
        
        for i in range(validation_minibatch_N):
            validation_X_minibatch[i][j] = validation_X[minibatch_index[i*minibatch_size + j]]
            validation_T_minibatch[i][j] = validation_T[minibatch_index[i*minibatch_size + j]]

    print("build SDA model")

    print("constructing the model...")
    classifer = StackedDenoisingAutoencoders(n_in, hidden_layer_sizes, n_out, rng)
    print("done.")

    print("pre-training the model...")
    classifer.pretrain(train_X_minibatch, minibatch_size, train_minibatch_N,
                       pretrain_epochs, pretrain_learning_rate, corruption_level)
    print("done.")

    print("fine-tuning the model...")
    for epoch in range(finetune_epochs):
        print("epoch=", epoch)
        for batch in range(validation_minibatch_N):
            classifer.finetune(validation_X_minibatch[batch],
                               validation_T_minibatch[batch],
                               minibatch_size, finetune_learning_rate)
        finetune_learning_rate *= 0.98
    print("done.")

    print("test the model")
    predicted_T = np.zeros((test_N, n_out))
    for i in range(test_N):
        predicted_T[i] = classifer.predict(test_X[i])

    print("evaluate the model")
    
    confusion_matrix = np.zeros((patterns, patterns))
    for i in range(test_N):
        predicted_ = np.where(predicted_T[i] == 1) # find the position of the value 1
        actual_ = np.where(test_T[i] == 1) # find the position of the value 1
        col = predicted_[0][0] # NOTE
        row = actual_[0][0] # NOTE
        confusion_matrix[row][col] += 1

    accuracy = 0.0
    precision = np.zeros(patterns)
    recall = np.zeros(patterns)
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

        if col_ > 0:
            precision[i] /= col_
        else:
            precision[i] = -0.99 # TODO Error
        if row_ > 0:
            recall[i] /= row_
        else:
            recall[i] = -0.99 # TODO Error

    accuracy /= test_N

    print("SDA model evaluation")
    print("Accuracy: " + str(accuracy * 100))
    print("Precision: ")
    for i in range(patterns):
        print("class: " + str(i+1) + ", precision: " + str(precision[i] * 100))
    print("Recall: ")
    for i in range(patterns):
        print("class: " + str(i+1) + ", recall: " + str(recall[i] * 100))


if __name__ == "__main__":
    test_stacked_denoising_autoencoders()