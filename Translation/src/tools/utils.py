"""
Utility functions for building neural networks in theano.
Most of these are taken from http://deeplearning.net/tutorial/.
"""

from __future__ import print_function

import os
import sys
import timeit
import gzip
import pickle

import numpy as np
import theano
import theano.tensor as T


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    train_set, valid_set, test_set = train_set[0], valid_set[0], test_set[0]
    return train_set, valid_set, test_set


def split_img(img):
    """
    Split mnist image

    Split an mnist example input (28x28 np.array of floats) into two
    called x and y, where x is the upper first 392=784/2 pixels and y
    the latter 392 pixels.

    :param img: binary numpy array of an mnist image
    :return: two half images for x and y
    """
    dim = img.shape[1]
    x = img[:, :dim]
    y = img[:, :-dim]
    return x, y


class HiddenLayer(object):
    def __init__(self, input_tensor, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu):
        """
        Hidden layer of a neural network.

        Args:
        input_tensor: a tensor as the input to the layer
        n_in: the number of features of the input tensor
        n_out: the number of features we are mapping to
        W: weight matrix
        b: bias term
        activation: activation function we are going to use
        """
        self.input_tensor = input_tensor
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh. This means that ReLU will be initialized by the same
        #        values as for tanh.
        if W is None:
            W_values = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input_tensor, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class EncoderMLP(object):
    def __init__(self, srng, x, n_in, n_hid, ldim):
        """
        Gaussian MLP that handles the encoding q_phi(z | x).

        Outputs a mean tensor mu and diagonal entries of the covariance matrix, sigma2.
        Args:
        rng: np random number generator
        srng: theano shared random number generator
        input_tensor: input tensor that we feed into the network
        n_in: number of features of the input vector
        n_hid: number of hidden units
        lat_dim: Latent dimension of the space, dictates size of mu and sigma2
        """
        # Layers
        self.hidden_layer = HiddenLayer(x, n_in, n_hid, activation=T.tanh)
        self.mu_layer = HiddenLayer(self.hidden_layer.output, n_hid, ldim, activation=None)
        self.logsigma2_layer = HiddenLayer(self.hidden_layer.output, n_hid, ldim, activation=None)
        # Tensors
        self.mu = self.mu_layer.output
        self.sigma2 = T.exp(self.logsigma2_layer.output)
        self.sigma = T.sqrt(self.sigma2)
        self.eps = srng.normal(self.mu.shape)
        self.output = self.mu + self.sigma * self.eps
        # Parameters
        self.params = self.hidden_layer.params + self.mu_layer.params + self.logsigma2_layer.params


class FactoredGaussianEncoderMLP(object):
    def __init__(self, srng, input_tensor, n_in, n_hid, ldim):
        """
        Gaussian dual MLP for encoding of form q_phi(z | x, y) = q(z | x)q(z | y)

        Keeps as parameters the mu and sigma of the x and y, and sample a z from
        the join distribution of these. We consider precision since this is easier
        work with than variances when calculating the joint distribution of two
        Gaussians.

        :param srng: theano random number generator
        :param x: input tensor for x in (x, y) input
        :param y: input tensor for y in (x, y) input
        :param nx_in: input dimension for x
        :param ny_in: input dimension for y
        :param n_hid: size of the hidden layer
        :param ldim: number of samples from z
        """
        # split tensor into 2
        x = input_tensor[:, :392]
        y = input_tensor[:, :-392]

        # X part of NN
        # Layers
        self.hidden_layer_x = HiddenLayer(x, n_in//2, n_hid, activation=T.tanh)
        self.mu_layer_x = HiddenLayer(self.hidden_layer_x.output, n_hid, ldim, activation=None)
        self.loglambda_layer_x = HiddenLayer(self.hidden_layer_x.output, n_hid, ldim, activation=None)
        # Tensors
        self.mu_x = self.mu_layer_x.output
        self.lambda_x = T.exp(self.loglambda_layer_x.output)

        # Y part of NN
        # Layers
        self.hidden_layer_y = HiddenLayer(y, n_in//2, n_hid, activation=T.tanh)
        self.mu_layer_y = HiddenLayer(self.hidden_layer_y.output, n_hid, ldim, activation=None)
        self.loglambda_layer_y = HiddenLayer(self.hidden_layer_y.output, n_hid, ldim, activation=None)
        # Tensors
        self.mu_y = self.mu_layer_y.output
        self.lambda_y = T.exp(self.loglambda_layer_y.output)

        # The distribution q(z | x, y) can be shown to be N(mu, lambda)
        # We have assumed diagonal covariance structure for q(z | x) and
        # q(z | y) which implies covariance structure for the precision as
        # well, hence we only need lambda to be a vector. We have that:
        # lambda = (lambda_x + lambda_y)
        # mu = (1/lambda)(lambda_x * mu_x + lambda_y * mu_y)
        self.lambda_xy = self.lambda_x + self.lambda_y
        self.mu_xy = (self.lambda_x * self.mu_x + self.lambda_y * self.mu_y) / self.lambda_xy

        # Sample from z
        self.eps = srng.normal(self.mu_xy.shape)
        self.sigma_xy = T.sqrt(1.0/self.lambda_xy)
        self.output = self.mu_xy + self.sigma_xy * self.eps
        # Fantasize z by masking x
        self.eps_mask = srng.normal(self.mu_x.shape)
        self.sigma_x = T.sqrt(1.0/self.lambda_x)
        self.output_mask = self.mu_x + self.sigma_x * self.eps_mask

        # Parameters
        self.params = self.hidden_layer_x.params + self.mu_layer_x.params + self.loglambda_layer_x.params
        self.params += self.hidden_layer_y.params + self.mu_layer_y.params + self.loglambda_layer_y.params


class DecoderMLP(object):
    def __init__(self, x, n_in, n_hid, n_out):
        """ Decoder MLP for getting the final output of the network

        Args:
        input_tensor: input tensor that we feed into the network
        n_in: number of features of the input vector
        n_hid: number of hidden units
        """
        self.hidden_layer = HiddenLayer(x, n_in, n_hid, activation=T.tanh)
        self.bern_layer = HiddenLayer(self.hidden_layer.output, n_hid, n_out, activation=T.nnet.sigmoid)
        self.params = self.hidden_layer.params + self.bern_layer.params
        self.bern = self.bern_layer.output


# Helper functions with regard to cost functions

def kl_unit_normal(mu, var):
    """
    Assuming normality of the target probability we have that since
    the latent prior is unit normal, from
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Kullback.E2.80.93Leibler_divergence_for_multivariate_normal_distributions
    and from AEVB paper page. 5. that
    D_KL(q_phi(z | x_i) || p_theta(z)) = 0.5 * (sum(var) + sum(mu**2) - sum(log(var)) - k)"""
    return 0.5 * (T.sum(var, axis=1) + T.sum(T.sqr(mu), axis=1) - T.sum(1 + T.log(var), axis=1))
