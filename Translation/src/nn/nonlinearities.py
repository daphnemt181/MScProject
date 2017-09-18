import theano.tensor as T
from lasagne.nonlinearities import elu


def elu_plus_one(x):

    return elu(x) + 1. + 1.e-5


def log_linear(x):

    return T.switch(T.lt(x, 0), -T.log(1. - x), x) + 1.

