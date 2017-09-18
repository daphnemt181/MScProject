"""
Recognition models for translation project
"""

import theano.tensor as T
import lasagne
from lasagne.layers import Conv1DLayer, DenseLayer, DimshuffleLayer, ElemwiseMergeLayer, ElemwiseSumLayer, Gate, \
    get_all_layers, get_all_param_values, get_all_params, get_output, InputLayer, LSTMLayer, NonlinearityLayer, \
    ReshapeLayer, set_all_param_values, SliceLayer
from lasagne.nonlinearities import elu, linear, tanh, sigmoid
import sys

sys.path.append("../nn")

from layers import DilatedConv1DLayer
from nonlinearities import elu_plus_one

class RecognitionModel(object):
    """Recognition model for translation using a factored gaussian

    We split up the variational equation such that it's gaussian
    and that q(z | x, y) = q(z | x)q(z | y)
    """

    def __init__(self, input_max_len_x, input_max_len_y, vocab_size, z_dim, z_dist):
        """
        :param input_max_len_x: (int) maximum length of input data (x)
        :param input_max_len_y: (int) maximum length of input data (y)
        :param input_vocab_size_x: (int) size of the vocabulary from input language x
        :param input_vocab_size_y: (int) size of the vocabulary from input language y
        :param z_dim: (int) latent dimensionality of z
        :param z_dist: (distribution) latent distribution (will define behaviour of this as well)
        NOTE: z_dim, z_dist have to be the same for recognition model
        and generative model by definition.
        """

        self.input_max_len_x = input_max_len_x
        self.input_max_len_y = input_max_len_y
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        self.z_dist = z_dist()

        # get mean and covariance vector for x and y
        self.mean_x, self.cov_x = self.nn_fn(input_max_len_x)
        self.mean_y, self.cov_y = self.nn_fn(input_max_len_y)

    def nn_fn(self, max_length=None):

        raise NotImplementedError()

    def get_q_params(self, x, y, x_embedded, y_embedded):
        """Given input x, y, get output mean and covariance

        :param x: (N * max(L) * D_x tensor) tensor input for x
        :param y: (N * max(L) * D_x tensor) tensor input for y

        :return mean_q: (N * dim(z) tensor) output mean tensor
        :return cov_q: (N * dim(z) tensor) output cov tensor"""
        # mean and covariance

        mask_x = T.switch(T.lt(x, 0), 0, 1)  # N * max(L)
        mask_y = T.switch(T.lt(y, 0), 0, 1)  # N * max(L)

        x_embedded *= T.shape_padright(mask_x)
        y_embedded *= T.shape_padright(mask_y)

        mean_x_ = get_output(self.mean_x, x_embedded)  # N * z_dim
        cov_x_ = get_output(self.cov_x, x_embedded)  # N * z_dim
        mean_y_ = get_output(self.mean_y, y_embedded)  # N * z_dim
        cov_y_ = get_output(self.cov_y, y_embedded)  # N * z_dim

        mean_q, cov_q = factored_gaussian_params_fn(mean_x_, cov_x_, mean_y_, cov_y_)

        return mean_q, cov_q

    def get_means_covs_translation(self, x, y, x_embedded, y_embedded, translation_source):
        if translation_source is 'x':
            mask = T.switch(T.lt(x, 0), 0, 1)  # N * max(L)
            x_embedded *= T.shape_padright(mask)
            means = get_output(self.mean_x, x_embedded)  # N * z_dim
            covs = get_output(self.cov_x, x_embedded)  # N * z_dim
        else:
            mask = T.switch(T.lt(y, 0), 0, 1)  # N * max(L)
            y_embedded *= T.shape_padright(mask)
            means = get_output(self.mean_y, y_embedded)  # N * z_dim
            covs = get_output(self.cov_y, y_embedded)  # N * z_dim

        return means, covs

    def get_samples(self, x, y, x_embedded, y_embedded, num_samples, means_only=False):
        """
        :param x: ((N * max(L) * D_x) tensor) input from x
        :param y: ((N * max(L) * D_y) tensor) input from y
        :param x_embedded: N * max(L) * D tensor
        :param y_embedded: N * max(L) * D tensor
        :param num_samples: (int) number of samples to sample from z
        :param means_only: (bool) flag for just using mean

        :return samples: (N * S * dim(z) tensor) latent samples from batch
        """

        # Use nn to get the parameters of q(z|x,y)
        means_q, covs_q = self.get_q_params(x, y, x_embedded, y_embedded)

        # if means_only then we give back the mean
        # else we sample it from the distribution
        if means_only:
            samples = T.tile(means_q, [num_samples] + [1]*(means_q.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.z_dist.get_samples(num_samples, [means_q, covs_q])  # (S*N) * dim(z)

        return samples

    def get_samples_translation(self, x, y, x_embedded, y_embedded, num_samples, means_only=False, translation_source='x'):
        """
        :param x: ((N * max(L) * D_x) tensor) input from x
        :param y: ((N * max(L) * D_y) tensor) input from y
        :param x_embedded: N * max(L) * D tensor
        :param y_embedded: N * max(L) * D tensor
        :param num_samples: (int) number of samples to sample from z
        :param means_only: (bool) flag for just using mean

        :return samples: (N * S * dim(z) tensor) latent samples from batch
        """

        means, covs = self.get_means_covs_translation(x, y, x_embedded, y_embedded, translation_source)

        # if means_only then we give back the mean
        # else we sample it from the distribution
        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.z_dist.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        return samples

    def log_q_z(self, z, x, y, x_embedded, y_embedded):
        """Log-probabilities of q(z | x, y) (may assume that q(z | x, y) = q(z | x)q(z | y)
        or just that we concatenate input)

        Calculate the log-probability of q(z | x) elementwise using the distribution of z | x
        which is assumed by the recognition model itself (most of the cases VAE). z | x is already
        sampled from before

        :param z: (N * z_dim tensor) tensor of the samples latents conditioned on x
        :param x: (N * max(L) * D_x tensor) tensor of the batch input
        :param y: (N * max(L) * D_y tensor) tensor of the batch input
        :param x_embedded: N * max(L) * D tensor
        :param y_embedded: N * max(L) * D tensor

        :return log_q_z: (theano symbolic function) symbolic function of the
        log-probabilities (N vector)
        """
        N = x.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        means, covs = self.get_q_params(x, y, x_embedded, y_embedded)

        means = T.tile(means, [S] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        covs = T.tile(covs, [S] + [1]*(means.ndim - 1))  # (S*N) * dim(z)

        # actual log-likelihoods
        log_q_z =  self.z_dist.log_density(z, [means, covs])  # N

        return log_q_z

    def log_q_z_translation(self, z, x, y, x_embedded, y_embedded, translation_source='x'):
        """Log-probabilities of q(z | x) or q(z | y)

        Calculate the log-probability of q(z | x) elementwise using the distribution of z | x
        which is assumed by the recognition model itself (most of the cases VAE). z | x is already
        sampled from before

        :param z: (N * z_dim tensor) tensor of the samples latents conditioned on x or y
        :param x: (N * max(L) * D_x tensor) tensor of the batch input
        :param y: (N * max(L) * D_x tensor) tensor of the batch input

        :return log_q_z: (theano symbolic function) symbolic function of the
        log-probabilities (N vector)
        """

        N = x.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        means, covs = self.get_means_covs_translation(x, y, x_embedded, y_embedded, translation_source)

        means = T.tile(means, [S] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        covs = T.tile(covs, [S] + [1]*(means.ndim - 1))  # (S*N) * dim(z)

        log_q_z = self.z_dist.log_density(z, [means, covs])

        return log_q_z

    def kl_std_gaussian(self, x, y, x_embedded, y_embedded, translation=False, translation_source=None):
        """
        :param X: N * max(L) * D tensor

        :return kl: N length vector
        """
        if not translation:
            means, covs = self.get_q_params(x, y, x_embedded, y_embedded)
        else:
            means, covs = self.get_means_covs_translation(x, y, x_embedded, y_embedded, translation_source)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return kl

    def get_params(self):
        """Get parameters, phi"""
        nn_params = get_all_params(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]), trainable=True)
        return nn_params

    def get_param_values(self):
        """Get parameters, phi, non-mutable"""
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]))
        return [nn_params_vals]

    def set_param_values(self, param_values):
        """Set parameters, phi"""
        [nn_params_vals] = param_values
        set_all_param_values(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]), nn_params_vals)


class RecMLP(RecognitionModel):

    def __init__(self, input_max_len_x, input_max_len_y, vocab_size, nn_kwargs, z_dim, z_dist):
        """
        :param input_max_len_x: (int) maximum length of input data (x)
        :param input_max_len_y: (int) maximum length of input data (y)
        :param input_vocab_size_x: (int) size of the vocabulary from input language x
        :param input_vocab_size_y: (int) size of the vocabulary from input language y
        :param nn_kwargs: (dict) dictionary of all of the needed kwargs for the
        neural network used by the recognition model (same for x and y)
        :param z_dim: (int) latent dimensionality of z
        :param z_dist: (distribution) latent distribution (will define behaviour of this as well)
        NOTE: z_dim, z_dist have to be the same for recognition model
        and generative model by definition.
        """

        self.nn_depth = nn_kwargs['depth']
        self.nn_hid_units = nn_kwargs['hid_units']
        self.nn_hid_nonlinearity = nn_kwargs['hid_nonlinearity']

        super(RecMLP, self).__init__(input_max_len_x, input_max_len_y, vocab_size, z_dim, z_dist)

        # get mean and covariance vector for x and y
        self.mean_x, self.cov_x = self.nn_fn(input_max_len_x)
        self.mean_y, self.cov_y = self.nn_fn(input_max_len_y)

    def nn_fn(self, max_length):
        """Build the theano tensor which represents the MLP

        Using the attributes of the class, build the theano graph
        using lasagne to get the neural networks which can
        then be passed into theano.function to get the values
        given input.

        :param max_length: (int) max length of the language we are using

        :return mean_nn_fn: (tensor) theano tensor specifying the mean
        :return cov_nn_fn: (tensor) theano tensor specifying the covariance"""

        l_in = InputLayer((None, max_length, self.vocab_size))
        l_current = l_in

        for h in range(self.nn_depth):
            l_h_x = DenseLayer(l_in, num_units=self.nn_hid_units, nonlinearity=None, b=None)
            l_h_h = DenseLayer(l_current, num_units=self.nn_hid_units, nonlinearity=None, b=None)
            l_current = NonlinearityLayer(ElemwiseSumLayer([l_h_x, l_h_h]), nonlinearity=self.nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_current, num_units=self.z_dim, nonlinearity=linear, b=None)
        cov_nn = DenseLayer(l_current, num_units=self.z_dim, nonlinearity=elu_plus_one, b=None)

        return mean_nn, cov_nn

class RecRNN(RecognitionModel):

    def __init__(self, input_max_len_x, input_max_len_y, vocab_size, nn_kwargs, z_dim, z_dist):
        """
        :param input_max_len_x: (int) maximum length of input data (x)
        :param input_max_len_y: (int) maximum length of input data (y)
        :param input_vocab_size_x: (int) size of the vocabulary from input language x
        :param input_vocab_size_y: (int) size of the vocabulary from input language y
        :param nn_kwargs: (dict) dictionary of all of the needed kwargs for the
        neural network used by the recognition model (same for x and y)
        :param z_dim: (int) latent dimensionality of z
        :param z_dist: (distribution) latent distribution (will define behaviour of this as well)
        NOTE: z_dim, z_dist have to be the same for recognition model
        and generative model by definition.
        """

        self.nn_rnn_hid_dim = nn_kwargs['rnn_hid_dim']
        self.nn_final_depth = nn_kwargs['final_depth']
        self.nn_final_hid_units = nn_kwargs['final_hid_units']
        self.nn_final_hid_nonlinearity = nn_kwargs['final_hid_nonlinearity']

        super(RecRNN, self).__init__(input_max_len_x, input_max_len_y, vocab_size, z_dim, z_dist)

        # get mean and covariance vector for x and y
        self.rnn_x = self.rnn_fn(self.input_max_len_x)
        self.rnn_y = self.rnn_fn(self.input_max_len_y)

    def rnn_fn(self, max_length):
        l_in = InputLayer((None, max_length, self.vocab_size))
        l_mask = InputLayer((None, max_length))
        l_final = LSTMLayer(l_in, num_units=self.nn_rnn_hid_dim, mask_input=l_mask, only_return_final=True)
        return l_final

    def nn_fn(self, max_length):
        """Build the theano tensor

        Using the attributes of the class, build the theano graph
        using lasagne to get the neural networks which can
        then be passed into theano.function to get the values
        given input.

        :return mean_nn_fn: (tensor) theano tensor specifying the mean
        :return cov_nn_fn: (tensor) theano tensor specifying the covariance"""

        l_in = InputLayer((None, self.nn_rnn_hid_dim))
        l_prev = l_in

        for h in range(self.nn_final_depth):
            l_prev = DenseLayer(l_prev, num_units=self.nn_final_hid_units, nonlinearity=self.nn_final_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)
        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_q_params(self, x, y, x_embedded, y_embedded):
        """Given input x, y, get output mean and covariance

        :param x: (N * max(L) * D_x tensor) tensor input for x
        :param y: (N * max(L) * D_x tensor) tensor input for y

        :return mean_q: (N * dim(z) tensor) output mean tensor
        :return cov_q: (N * dim(z) tensor) output cov tensor"""
        # mean and covariance

        mask_x = T.switch(T.lt(x, 0), 0, 1)  # N * max(L)
        mask_y = T.switch(T.lt(y, 0), 0, 1)  # N * max(L)

        hid_x = self.rnn_x.get_output_for([x_embedded, mask_x])  # N * dim(z)
        hid_y = self.rnn_y.get_output_for([y_embedded, mask_y])  # N * dim(z)

        mean_x_ = get_output(self.mean_x, hid_x)  # N * z_dim
        cov_x_ = get_output(self.cov_x, hid_x)  # N * z_dim
        mean_y_ = get_output(self.mean_y, hid_y)  # N * z_dim
        cov_y_ = get_output(self.cov_y, hid_y)  # N * z_dim

        mean_q, cov_q = factored_gaussian_params_fn(mean_x_, cov_x_, mean_y_, cov_y_)

        return mean_q, cov_q

    def get_means_covs_translation(self, x, y, x_embedded, y_embedded, translation_source):
        if translation_source is 'x':
            mask = T.switch(T.lt(x, 0), 0, 1)  # N * max(L)
            hid = self.rnn_x.get_output_for([x_embedded, mask])  # N * dim(z)
            means = get_output(self.mean_x, hid)  # N * z_dim
            covs = get_output(self.cov_x, hid)  # N * z_dim
        else:
            mask = T.switch(T.lt(y, 0), 0, 1)  # N * max(L)
            hid = self.rnn_y.get_output_for([y_embedded, mask])  # N * dim(z)
            means = get_output(self.mean_y, hid)  # N * z_dim
            covs = get_output(self.cov_y, hid)  # N * z_dim

        return means, covs

    def get_params(self):
        """Get parameters, phi"""
        rnn_params = get_all_params(get_all_layers([self.rnn_x, self.rnn_y]), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):
        """Get parameters, phi, non-mutable"""
        rnn_params_vals = get_all_param_values(get_all_layers([self.rnn_x, self.rnn_y]))
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):
        """Set parameters, phi"""
        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(get_all_layers([self.rnn_x, self.rnn_y]), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]), nn_params_vals)


class RecWaveNetText(RecognitionModel):

    def __init__(self, input_max_len_x, input_max_len_y, vocab_size, nn_kwargs, z_dim, z_dist):
        """
        :param input_max_len_x: (int) maximum length of input data (x)
        :param input_max_len_y: (int) maximum length of input data (y)
        :param input_vocab_size_x: (int) size of the vocabulary from input language x
        :param input_vocab_size_y: (int) size of the vocabulary from input language y
        :param nn_kwargs: (dict) dictionary of all of the needed kwargs for the
        neural network used by the recognition model (same for x and y)
        :param z_dim: (int) latent dimensionality of z
        :param z_dist: (distribution) latent distribution (will define behaviour of this as well)
        NOTE: z_dim, z_dist have to be the same for recognition model
        and generative model by definition.
        """
        self.nn_dilations = nn_kwargs['dilations']
        self.nn_dilation_channels = nn_kwargs['dilation_channels']
        self.nn_residual_channels = nn_kwargs['residual_channels']
        self.nn_final_depth = nn_kwargs['final_depth']
        self.nn_final_hid_units = nn_kwargs['final_hid_units']
        self.nn_final_hid_nonlinearity = nn_kwargs['final_hid_nonlinearity']

        super(RecWaveNetText, self).__init__(input_max_len_x, input_max_len_y, vocab_size, z_dim, z_dist)

        self.cnn_x = self.cnn_fn(input_max_len_x)
        self.cnn_y = self.cnn_fn(input_max_len_y)

    def cnn_fn(self, max_length):
        """Build the theano tensor

        Using the attributes of the class, build the theano graph
        using lasagne to get the neural networks which can
        then be passed into theano.function to get the values
        given input.

        :param max_len: (int) max length of the language we are using

        :return l_final: (tensor) theano tensor specifying the output of the cnn"""

        l_in = InputLayer((None, max_length, self.vocab_size))
        l_in_T = DimshuffleLayer(l_in, (0, 2, 1))
        l_causal_conv = DilatedConv1DLayer(l_in_T, num_filters=self.nn_residual_channels, dilation=1, nonlinearity=None)
        l_prev = l_causal_conv

        skip_layers = []

        for h in range(len(self.nn_dilations)):

            l_filter = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels, dilation=self.nn_dilations[h], nonlinearity=tanh)

            l_gate = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels, dilation=self.nn_dilations[h], nonlinearity=sigmoid)

            l_merge = ElemwiseMergeLayer([l_filter, l_gate], merge_function=T.mul)

            l_dense = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None)

            l_residual = ElemwiseSumLayer([l_prev, l_dense])

            l_skip = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None)

            skip_layers.append(l_skip)

            l_prev = l_residual

        l_skip_sum = NonlinearityLayer(ElemwiseSumLayer(skip_layers), nonlinearity=elu)

        l_final = DimshuffleLayer(l_skip_sum, (0, 2, 1))

        return l_final

    def nn_fn(self, max_length):
        """Build the theano tensor

        Using the attributes of the class, build the theano graph
        using lasagne to get the neural networks which can
        then be passed into theano.function to get the values
        given input.

        :return mean_nn_fn: (tensor) theano tensor specifying the mean
        :return cov_nn_fn: (tensor) theano tensor specifying the covariance"""

        l_in = InputLayer((None, self.nn_residual_channels))
        l_prev = l_in

        if self.nn_final_depth > 0:
            for h in range(self.nn_final_depth):
                l_prev = DenseLayer(l_prev, num_units=self.nn_final_hid_units, nonlinearity=self.nn_final_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)
        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_q_params(self, x, y, x_embedded, y_embedded):
        """Given input x, y, get output mean and covariance

        :param x: (N * max(L) * D_x tensor) tensor input for x
        :param y: (N * max(L) * D_x tensor) tensor input for y

        :return mean_q: (N * dim(z) tensor) output mean tensor
        :return cov_q: (N * dim(z) tensor) output cov tensor"""
        # mean and covariance

        mask_x = T.switch(T.lt(x, 0), 0, 1)  # N * max(L)
        mask_y = T.switch(T.lt(y, 0), 0, 1)  # N * max(L)

        x_embedded *= T.shape_padright(mask_x)
        y_embedded *= T.shape_padright(mask_y)

        hid_x = get_output(self.cnn_x, x_embedded)  # N * max(L) * dim(z)
        hid_y = get_output(self.cnn_y, y_embedded)  # N * max(L) * dim(z)

        L_x = T.cast(T.sum(mask_x, axis=1) - 1., 'int32')  # N
        L_y = T.cast(T.sum(mask_y, axis=1) - 1., 'int32')  # N

        hid_final_x = hid_x[T.arange(x.shape[0]), L_x]
        hid_final_y = hid_y[T.arange(y.shape[0]), L_y]

        mean_x_ = get_output(self.mean_x, hid_final_x)  # N * z_dim
        cov_x_ = get_output(self.cov_x, hid_final_x)  # N * z_dim
        mean_y_ = get_output(self.mean_y, hid_final_y)  # N * z_dim
        cov_y_ = get_output(self.cov_y, hid_final_y)  # N * z_dim

        mean_q, cov_q = factored_gaussian_params_fn(mean_x_, cov_x_, mean_y_, cov_y_)

        return mean_q, cov_q

    def get_means_covs_translation(self, x, y, x_embedded, y_embedded, translation_source):
        if translation_source is 'x':
            mask = T.switch(T.lt(x, 0), 0, 1)  # N * max(L)
            x_embedded *= T.shape_padright(mask)
            hid = get_output(self.cnn_x, x_embedded)  # N * max(L) * dim(z)
            L = T.cast(T.sum(mask, axis=1) - 1., 'int32')  # N
            hid_final = hid[T.arange(x.shape[0]), L]
            means = get_output(self.mean_x, hid_final)  # N * z_dim
            covs = get_output(self.cov_x, hid_final)  # N * z_dim
        else:
            mask = T.switch(T.lt(y, 0), 0, 1)  # N * max(L)
            y_embedded *= T.shape_padright(mask)
            hid = get_output(self.cnn_y, y_embedded)  # N * max(L) * dim(z)
            L = T.cast(T.sum(mask, axis=1) - 1., 'int32')  # N
            hid_final = hid[T.arange(y.shape[0]), L]
            means = get_output(self.mean_y, hid_final)  # N * z_dim
            covs = get_output(self.cov_y, hid_final)  # N * z_dim

        return means, covs

    def get_params(self):
        """Get parameters, phi"""
        cnn_params = get_all_params(get_all_layers([self.cnn_x, self.cnn_y]), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]), trainable=True)

        return cnn_params + nn_params

    def get_param_values(self):
        """Get parameters, phi, non-mutable"""
        cnn_params_vals = get_all_param_values(get_all_layers([self.cnn_x, self.cnn_y]))
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]))

        return [cnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):
        """Set parameters, phi"""
        [cnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(get_all_layers([self.cnn_x, self.cnn_y]), cnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_x, self.cov_x, self.mean_y, self.cov_y]), nn_params_vals)

# functions

def factored_gaussian_params_fn(mean_x, cov_x, mean_y, cov_y):
    """calculate the parameters of the gaussian q(z | y)q(z | x)

    Given that the form of the gaussian q(z | x, y) \propto q(z | x)q(z | y)
    is another gaussian with closed form covariance and mean vectors, we
    can easily get it from the output of the two networks

    :param mean_x: ((N * z_dim) tensor) mean from input x
    :param cov_x: ((N * z_dim) tensor) covariance diagonal from input x
    :param mean_y: ((N * z_dim) tensor) mean from input y
    :param cov_y: ((N * z_dim) tensor) covariance diagonal from input y

    :return mean_q: ((N * z_dim) tensor) mean of the q-gaussian
    :return cov_q: ((N * z_sim) tensor) covariance of the q-gaussian"""
    cov_q = 1.0/((1.0/cov_x) + (1.0/cov_y))
    mean_q = ((1.0/cov_x) * mean_x + (1.0/cov_y) * mean_y) * cov_q

    return mean_q, cov_q
