import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams()


class GaussianDiagonal(object):
    """Class for the multivariate Gaussian distribution with diagonal covariance structure"""

    def density(self, obs, params=None):
        """Get the density of a point under the distribution

        :param obs: observation
        :param params: list of parameters mu, covs
        :return: density"""

        if params is not None:
            means = params[0]
            covs = params[1]  # covs should be of the same size as obs, i.e. only diagonal Gaussians are permitted.
        else:
            means = T.zeros_like(obs)
            covs = T.ones_like(obs)

        normalisers = ((2. * np.pi * covs) ** (-0.5))
        densities = T.exp(-((obs - means) ** 2) / (2. * covs))

        return normalisers * densities

    def log_density(self, obs, params=None, sum_trailing_axes=True):
        """Get the log-density of a point under the distribution

        :param obs: observation
        :param params: list of parameters mu, covs
        :return: log-density"""
        if params is not None:
            means = params[0]
            covs = params[1]  # covs should be of the same size as obs, i.e. only diagonal Gaussians are permitted.
        else:
            means = T.zeros_like(obs)
            covs = T.ones_like(obs)

        ndim = obs.ndim

        if sum_trailing_axes:
            log_normalisers = - 0.5 * T.sum(T.log(2. * np.pi * covs), axis=range(1, ndim))
            log_densities = - 0.5 * T.sum(((obs - means)**2) / covs, axis=range(1, ndim))
        else:
            log_normalisers = - 0.5 * T.log(2. * np.pi * covs)
            log_densities = - 0.5 * (((obs - means)**2) / covs)

        return log_normalisers + log_densities

    def get_samples(self, num_samples, params=None, dims=None):
        """Generate samples from the distribution

        :param num_samples: number of samples to generate
        :param params: parameters of the Guassian
        :param dims: dimensions in case we don't pass parameters
        :return samples: generated samples"""
        if params is not None:
            means = params[0]
            covs = params[1]
        else:
            means = T.zeros(dims)
            covs = T.ones(dims)

        # tile so we just have to sample from a num_samples * dims
        # Gaussian instead of sampling from a gaussian num_samples times
        means = T.tile(means, [num_samples] + [1]*(means.ndim - 1))
        covs = T.tile(covs, [num_samples] + [1]*(covs.ndim - 1))

        # unit normal e
        e = random.normal(means.shape)
        samples = means + (T.sqrt(covs) * e)

        return samples


class Categorical(object):

    def log_density(self, obs, params):
        """
        :param obs: N * max(L) * D matrix
        :param params: [probs: N * max(L) * D matrix]

        :return log_densities: N length vector
        """
        probs = params[0]
        probs += T.cast(1.e-15, 'float32')

        log_densities = T.sum(obs * T.log(probs), axis=(1, 2))

        return log_densities

    def get_samples(self, params):
        """
        :param params: [probs: N * max(L) * D tensor]
        :return samples: N * max(L) * D tensor
        """

        probs = params[0]

        samples_flat = random.multinomial(pvals=probs.reshape((probs.shape[0] * probs.shape[1], probs.shape[2])))

        samples = samples_flat.reshape(probs.shape)

        return samples


class Exponential(object):

    def log_density(self, obs, params=None):
        if params is not None:
            lambdas = params[0]
        else:
            lambdas = T.ones_like(obs)

        ndim = obs.ndim
        log_density = T.sum(T.log(lambdas) - (lambdas * obs), axis=range(1, ndim))

        return log_density

    def get_samples(self, num_samples, params=None, dims=None):
        if params is not None:
            lambdas = params[0]
        else:
            lambdas = T.ones(dims)

        lambdas = T.tile(lambdas, [num_samples] + [1]*(lambdas.ndim - 1))
        e = random.uniform(size=lambdas.shape)
        samples = - T.log(e) / lambdas

        return samples
