import numpy as np

from denoising_autoencoders import DenoisingAutoencoders
from dataset import make_binomial_dataset


def test_denoising_autoencoders():

    print("set configs")

    rng = np.random.RandomState(1234)

    train_N_each = 200
    test_N_each = 2
    n_visible_each = 4
    p_noise_training = 0.05
    p_noise_test = 0.25

    patterns = 3
    
    train_N = train_N_each * patterns
    test_N = test_N_each * patterns

    n_visible = n_visible_each * patterns
    n_hidden = 6
    corruption_level = 0.3

    epochs = 100 # 1000: slow
    learning_rate = 0.2
    minibatch_size = 10
    minibatch_N = int(train_N / minibatch_size)

    print("make training dataset")
    train_X = np.zeros((train_N, n_visible))
    make_binomial_dataset(train_X, patterns, train_N_each, n_visible_each, p_noise_training, rng)
    
    print("make test dataset")
    test_X = np.zeros((test_N, n_visible))
    make_binomial_dataset(test_X, patterns, test_N_each, n_visible_each, p_noise_test, rng)

    print("make minibatches")
    minibatch_train_X = np.zeros((minibatch_N, minibatch_size, n_visible))
    minibatch_index = [i for i in range(train_N)]
    rng.shuffle(minibatch_index)
    for i in range(minibatch_N):
        for j in range(minibatch_size):
            index = minibatch_index[i * minibatch_size + j]
            minibatch_train_X[i][j] = train_X[index]
    
    print("construct DA model")
    nn = DenoisingAutoencoders(n_visible, n_hidden, W=None, h_bias=None, v_bias=None, rng=rng)

    print("train")
    for epoch in range(epochs):
        print("epoch=", epoch)
        for batch in range(minibatch_N):
            nn.train(minibatch_train_X[batch], minibatch_size, learning_rate, corruption_level)

    print("test (reconstruct noised data)")
    reconstructed_X = np.zeros((test_N, n_visible))
    for i in range(test_N):
        reconstructed_X[i] = nn.reconstruct(test_X[i])

    print("evaluate DA model reconstruction")
    for pattern in range(patterns):
        print("class:", pattern + 1)

        for n in range(test_N_each):
            n_ = pattern * test_N_each + n
            print(test_X[n_], "->")
            print("[")
            for i in range(n_visible):
                print(str(reconstructed_X[n_][i]) + ", ")
            print("]")
            
        print("-------------")


if __name__ == "__main__":
    test_denoising_autoencoders()