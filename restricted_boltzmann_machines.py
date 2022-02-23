import numpy as np

from activation_function import sigmoid


class RestrictedBoltzmannMachines(object):

    def __init__(self, n_visible: int, n_hidden: int, W, h_bias, v_bias, rng):

        if rng is None:
            rng = np.random.RandomState(1234)

        # make initial weights by the uniform distribution
        if W is None:
            W = np.zeros((n_hidden, n_visible))
            w_ = 1. / float(n_visible)

            for j in range(n_hidden):
                for i in range(n_visible):
                    # W: (n_hidden, n_visible)
                    W[j][i] = rng.uniform(-w_, w_, 1)
        
        # initial biases of the hidden layer are 0
        if h_bias is None:
            h_bias = np.zeros(n_hidden)

            for j in range(n_hidden):
                h_bias[j] = 0.

        # initial biases of the visible layer are 0
        if v_bias is None:
            v_bias = np.zeros(n_visible)

            for j in range(n_visible):
                v_bias[j] = 0.

        self.rng = rng
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = W
        self.h_bias = h_bias
        self.v_bias = v_bias

    def prop_up(self, v, w, bias: float) -> float:

        pre_activation = 0.
        for i in range(self.n_visible):
            pre_activation += w[i] * v[i]
        pre_activation += bias

        return sigmoid(pre_activation)

    def prop_down(self, h, i: int, bias: float) -> float:

        pre_activation = 0.
        for j in range(self.n_hidden):
            pre_activation += self.W[j][i] * h[j]
        pre_activation += bias

        return sigmoid(pre_activation)

    def sample_h_given_v(self, v0_samples, means, samples):
        
        for j in range(self.n_hidden):
            means[j] = self.prop_up(v0_samples, self.W[j], self.h_bias[j])
            samples[j] = self.rng.binomial(n=1, p=means[j])

    def sample_v_given_h(self, h0_samples, means, samples):
        
        for i in range(self.n_visible):
            means[i] = self.prop_down(h0_samples, i, self.v_bias[i])
            samples[i] = self.rng.binomial(n=1, p=means[i])

    def gibbs_hvh(self, h0_samples, nv_means, nv_samples, nh_means, nh_samples):
        
        self.sample_v_given_h(h0_samples, nv_means, nv_samples)
        self.sample_h_given_v(nv_samples, nh_means, nh_samples)


    def contrastive_divergence(self, X, minibatch_size: int, learning_rate: float, k: int):

        grad_W = np.zeros((self.n_hidden, self.n_visible))
        grad_h_bisas = np.zeros(self.n_hidden)
        grad_v_bias = np.zeros(self.n_visible)

        # train with minibatches
        for n in range(minibatch_size):

            ph_mean = np.zeros(self.n_hidden)
            ph_sample = np.zeros(self.n_hidden)

            nv_means = np.zeros(self.n_visible)
            nv_samples = np.zeros(self.n_visible)
            
            nh_means = np.zeros(self.n_hidden)
            nh_samples = np.zeros(self.n_hidden)

            # CD-k : CD-1 is enough for sampling (i.e. k=1)
            self.sample_h_given_v(X[n], ph_mean, ph_sample)

            for step in range(k):
                # Gibbs sampling
                if step == 0: # use ph_sample
                    self.gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples)
                else: # use nh_samples
                    self.gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples)

            # calculate gradients
            for j in range(self.n_hidden):
                for i in range(self.n_visible):
                    grad_W[j][i] += ph_mean[j] * X[n][i] - nh_means[j] * nv_samples[i]

                grad_h_bisas[j] += ph_mean[j] - nh_means[j]

            for i in range(self.n_visible):
                grad_v_bias[i] += X[n][i] - nv_samples[i]

            # update params (gradient descent)
            for j in range(self.n_hidden):
                for i in range(self.n_visible):
                    self.W[j][i] += learning_rate * grad_W[j][i] / minibatch_size

                self.h_bias[j] += learning_rate * grad_h_bisas[j] / minibatch_size

            for i in range(self.n_visible):
                self.v_bias[i] += learning_rate * grad_v_bias[i] / minibatch_size

    def reconstruct(self, v):
        
        h = np.zeros(self.n_hidden)
        # hidden layer calculation
        for j in range(self.n_hidden):
            h[j] = self.prop_up(v, self.W[j], self.h_bias[j])

        x = np.zeros(self.n_visible)
        # visible layer calculation
        for i in range(self.n_visible):
            pre_activation = 0.
            for j in range(self.n_hidden):
                pre_activation += self.W[j][i] * h[j]
            pre_activation += self.v_bias[i]

            x[i] = sigmoid(pre_activation)

        return x