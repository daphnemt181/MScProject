import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import Conv1DLayer, DenseLayer, DimshuffleLayer, ElemwiseMergeLayer, ElemwiseSumLayer, Gate, \
    get_all_layers, get_all_param_values, get_all_params, get_output, InputLayer, LSTMLayer, NonlinearityLayer, \
    ReshapeLayer, set_all_param_values, SliceLayer
from lasagne.nonlinearities import elu, linear, tanh, sigmoid
from nn.layers import DilatedConv1DLayer
from nn.nonlinearities import elu_plus_one


class RecModel(object):

    def __init__(self, z_dim, max_length, vocab_size, dist_z):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.dist_z = dist_z()

        self.mean_nn, self.cov_nn = self.nn_fn()

    def nn_fn(self):

        raise NotImplementedError()

    def get_means_and_covs(self, X, X_embedded):

        mask = T.switch(T.lt(X, 0), 0, 1)  # N * max(L)

        X_exmbedded *= T.shape_padright(mask)

        means = get_output(self.mean_nn, X_embedded)  # N * dim(z)
        covs = get_output(self.cov_nn, X_embedded)  # N * dim(z)

        return means, covs

    def get_samples(self, X, X_embedded, num_samples, means_only=False):
        """
        :param X: N * max(L) matrix
        :param X_embedded: N * max(L) * D tensor
        :param num_samples: int
        :param means_only: bool

        :return samples: (S*N) * dim(z) matrix
        """

        means, covs = self.get_means_and_covs(X, X_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        return samples

    def log_q_z(self, z, X, X_embedded):
        """
        :param z: (S*N) * dim(z) matrix
        :param X: N * max(L) * D tensor

        :return:
        """

        N = X.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        means, covs = self.get_means_and_covs(X, X_embedded)

        means = T.tile(means, [S] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        covs = T.tile(covs, [S] + [1]*(means.ndim - 1))  # (S*N) * dim(z)

        return self.dist_z.log_density(z, [means, covs])

    def kl_std_gaussian(self, X, X_embedded):
        """
        :param X: N * max(L) * D tensor

        :return kl: N length vector
        """

        means, covs = self.get_means_and_covs(X, X_embedded)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return kl

    def get_params(self):

        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecMLP(RecModel):

    def __init__(self, z_dim, max_length, vocab_size, dist_z, nn_kwargs):

        self.nn_depth = nn_kwargs['depth']
        self.nn_hid_units = nn_kwargs['hid_units']
        self.nn_hid_nonlinearity = nn_kwargs['hid_nonlinearity']

        super(RecMLP, self).__init__(z_dim, max_length, vocab_size, dist_z)

    def nn_fn(self):

        l_in = InputLayer((None, self.max_length, self.vocab_size))

        l_current = l_in

        for h in range(self.nn_depth):

            l_h_x = DenseLayer(l_in, num_units=self.nn_hid_units, nonlinearity=None, b=None)
            l_h_h = DenseLayer(l_current, num_units=self.nn_hid_units, nonlinearity=None, b=None)

            l_current = NonlinearityLayer(ElemwiseSumLayer([l_h_x, l_h_h]), nonlinearity=self.nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_current, num_units=self.z_dim, nonlinearity=linear, b=None)

        cov_nn = DenseLayer(l_current, num_units=self.z_dim, nonlinearity=elu_plus_one, b=None)

        return mean_nn, cov_nn


class RecRNN(RecModel):

    def __init__(self, z_dim, max_length, vocab_size, dist_z, nn_kwargs):

        self.nn_rnn_hid_dim = nn_kwargs['rnn_hid_dim']
        self.nn_final_depth = nn_kwargs['final_depth']
        self.nn_final_hid_units = nn_kwargs['final_hid_units']
        self.nn_final_hid_nonlinearity = nn_kwargs['final_hid_nonlinearity']

        super(RecRNN, self).__init__(z_dim, max_length, vocab_size, dist_z)

        self.rnn = self.rnn_fn()

    def rnn_fn(self):

        l_in = InputLayer((None, self.max_length, self.vocab_size))

        l_mask = InputLayer((None, self.max_length))

        l_final = LSTMLayer(l_in, num_units=self.nn_rnn_hid_dim, mask_input=l_mask, only_return_final=True)

        return l_final

    def nn_fn(self):

        l_in = InputLayer((None, self.nn_rnn_hid_dim))

        l_prev = l_in

        for h in range(self.nn_final_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_final_hid_units, nonlinearity=self.nn_final_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_means_and_covs(self, X, X_embedded):

        mask = T.switch(T.lt(X, 0), 0, 1)  # N * max(L)
        hid = self.rnn.get_output_for([X_embedded, mask])  # N * dim(z)

        means = get_output(self.mean_nn, hid)  # N * dim(z)
        covs = get_output(self.cov_nn, hid)  # N * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(self.rnn, trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(self.rnn)
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(self.rnn, rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecWaveNetText(RecModel):

    def __init__(self, z_dim, max_length, vocab_size, dist_z, nn_kwargs):

        self.nn_dilations = nn_kwargs['dilations']
        self.nn_dilation_channels = nn_kwargs['dilation_channels']
        self.nn_residual_channels = nn_kwargs['residual_channels']
        self.nn_final_depth = nn_kwargs['final_depth']
        self.nn_final_hid_units = nn_kwargs['final_hid_units']
        self.nn_final_hid_nonlinearity = nn_kwargs['final_hid_nonlinearity']

        super(RecWaveNetText, self).__init__(z_dim, max_length, vocab_size, dist_z)

        self.cnn = self.cnn_fn()

    def cnn_fn(self):

        l_in = InputLayer((None, self.max_length, self.vocab_size))

        l_in_T = DimshuffleLayer(l_in, (0, 2, 1))

        l_causal_conv = DilatedConv1DLayer(l_in_T, num_filters=self.nn_residual_channels, dilation=1, nonlinearity=None)

        l_prev = l_causal_conv

        skip_layers = []

        for h in range(len(self.nn_dilations)):

            l_filter = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels,
                                          dilation=self.nn_dilations[h], nonlinearity=tanh)

            l_gate = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels, dilation=self.nn_dilations[h],
                                        nonlinearity=sigmoid)

            l_merge = ElemwiseMergeLayer([l_filter, l_gate], merge_function=T.mul)

            l_dense = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None)

            l_residual = ElemwiseSumLayer([l_prev, l_dense])

            l_skip = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None)

            skip_layers.append(l_skip)

            l_prev = l_residual

        l_skip_sum = NonlinearityLayer(ElemwiseSumLayer(skip_layers), nonlinearity=elu)

        l_final = DimshuffleLayer(l_skip_sum, (0, 2, 1))

        return l_final

    def nn_fn(self):

        l_in = InputLayer((None, self.nn_residual_channels))

        l_prev = l_in

        if self.nn_final_depth > 0:

            for h in range(self.nn_final_depth):

                l_prev = DenseLayer(l_prev, num_units=self.nn_final_hid_units,
                                    nonlinearity=self.nn_final_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_means_and_covs(self, X, X_embedded):

        mask = T.switch(T.lt(X, 0), 0, 1)  # N * max(L)

        X_embedded *= T.shape_padright(mask)

        hid = get_output(self.cnn, X_embedded)  # N * max(L) * dim(z)

        L = T.cast(T.sum(mask, axis=1) - 1., 'int32')  # N

        hid_final = hid[T.arange(X.shape[0]), L]

        means = get_output(self.mean_nn, hid_final)  # N * dim(z)
        covs = get_output(self.cov_nn, hid_final)  # N * dim(z)

        return means, covs

    def get_params(self):

        cnn_params = get_all_params(self.cnn, trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return cnn_params + nn_params

    def get_param_values(self):

        cnn_params_vals = get_all_param_values(self.cnn)
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [cnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [cnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(self.cnn, cnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecWaveNetTextMultipleZ(RecWaveNetText):

    def __init__(self, z_dim, max_length, vocab_size, dist_z, nn_kwargs):

        super(RecWaveNetTextMultipleZ, self).__init__(z_dim, max_length, vocab_size, dist_z, nn_kwargs)

    def nn_fn(self):

        l_in = InputLayer((None, self.vocab_size, self.max_length))

        l_causal_conv = DilatedConv1DLayer(l_in, num_filters=self.nn_residual_channels, dilation=1, nonlinearity=None)

        l_prev = l_causal_conv

        skip_layers = []

        for h in range(len(self.nn_dilations)):

            l_filter = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels,
                                          dilation=self.nn_dilations[h], nonlinearity=tanh)

            l_gate = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels, dilation=self.nn_dilations[h],
                                        nonlinearity=sigmoid)

            l_merge = ElemwiseMergeLayer([l_filter, l_gate], merge_function=T.mul)

            l_dense = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None)

            l_residual = ElemwiseSumLayer([l_prev, l_dense])

            l_skip = Conv1DLayer(l_merge, num_filters=self.z_dim, filter_size=1, nonlinearity=None)

            skip_layers.append(l_skip)

            l_prev = l_residual

        l_skip_sum = NonlinearityLayer(ElemwiseSumLayer(skip_layers), nonlinearity=elu)

        l_prev = l_skip_sum

        for h in range(self.nn_final_depth):

            l_prev = Conv1DLayer(l_prev, num_filters=self.z_dim, filter_size=1,
                                 nonlinearity=self.nn_final_hid_nonlinearity)

        l_mean = DimshuffleLayer(Conv1DLayer(l_prev, num_filters=self.z_dim, filter_size=1, nonlinearity=linear),
                                 (0, 2, 1))

        l_cov = DimshuffleLayer(Conv1DLayer(l_prev, num_filters=self.z_dim, filter_size=1, nonlinearity=elu_plus_one),
                                (0, 2, 1))

        return l_mean, l_cov


class RecWaveNetTextMixtureZ(RecWaveNetText):

    def __init__(self, num_mixtures, z_dim, max_length, vocab_size, dist_mu, dist_sigma, dist_z, nn_kwargs):

        super(RecWaveNetTextMixtureZ, self).__init__(z_dim, max_length, vocab_size, dist_z, nn_kwargs)

        self.num_mixtures = num_mixtures

        self.dist_mu = dist_mu()
        self.dist_sigma = dist_sigma()

        self.means_mu = theano.shared(np.float32(np.random.normal(size=(self.num_mixtures, self.z_dim))))
        self.covs_mu_exps = theano.shared(np.float32(np.random.normal(size=(self.num_mixtures, self.z_dim))))

        self.lambdas_sigma_exps = theano.shared(np.float32(np.random.normal(size=(self.num_mixtures, self.z_dim))))

        self.mixture_params = [self.means_mu, self.covs_mu_exps, self.lambdas_sigma_exps]

    def get_samples(self, X, num_samples, means_only=False):
        """
        :param X: N * max(L) * D tensor
        :param num_samples: int
        :param means_only: bool

        :return samples: (S*N) * dim(z) matrix
        """

        means_z = get_output(self.mean_nn, X.dimshuffle((0, 2, 1)))  # N * dim(z)
        covs_z = get_output(self.cov_nn, X.dimshuffle((0, 2, 1)))  # N * dim(z)

        if means_only:
            samples_mu = self.means_mu  # K * dim(z)
            samples_sigma = 1. / T.exp(self.lambdas_sigma_exps)  # K * dim(z)
            samples_z = T.tile(means_z, (num_samples, 1))  # (S*N) * dim(z)
        else:
            samples_mu = self.dist_mu.get_samples(1, [self.means_mu, T.exp(self.covs_mu_exps)])  # K * dim(z)
            samples_sigma = self.dist_sigma.get_samples(1, [T.exp(self.lambdas_sigma_exps)])  # K * dim(z)
            samples_z = self.dist_z.get_samples(num_samples, [means_z, covs_z])  # (S*N) * dim(z)

        return samples_mu, samples_sigma, samples_z

    def log_q_mu(self, mu):

        log_q_mu = T.sum(self.dist_mu.log_density(mu, [self.means_mu, T.exp(self.covs_mu_exps)]))

        return log_q_mu

    def log_q_sigma(self, sigma):

        log_q_sigma = T.sum(self.dist_sigma.log_density(sigma, [T.exp(self.lambdas_sigma_exps)]))

        return log_q_sigma

    def get_params(self):

        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return nn_params + self.mixture_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        mixture_params_vals = [p.get_value() for p in self.mixture_params]

        return [nn_params_vals, mixture_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals, mixture_params_vals] = param_values

        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)

        for i in range(len(self.mixture_params)):
            self.mixture_params[i].set_value(mixture_params_vals[i])
