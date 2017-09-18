"""
Factored Gaussian for semi-supervised case.
"""

from __future__ import print_function

import argparse
import os
import sys
import timeit
import pickle
import time
import datetime
from pprint import pprint

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne.updates import adam, sgd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '../tools')
sys.path.insert(0, '../../')

import utils
from SETTINGS import PATHS

class FactoredGaussian(object):
    def __init__(self,
                 learning_rate,
                 hdim,
                 ldim,
                 input_tensor,
                 mask,
                 n_in,
                 n_out):
        # Set variables
        self.learning_rate = learning_rate
        self.hdim = hdim
        self.ldim = ldim
        self.input_tensor = input_tensor
        self.mask = mask
        self.n_in = n_in
        self.n_out = n_out
        # Build the network
        # We now have two heads and two tails
        self.srng = RandomStreams(seed=234)
        # Encoder part (phi)
        self.encoder = utils.FactoredGaussianEncoderMLP(self.srng,
                                                        self.input_tensor,
                                                        self.n_in,
                                                        self.hdim,
                                                        self.ldim)
        # Conditional of encoder output
        encoder_output = theano.ifelse.ifelse(self.mask, self.encoder.output_mask, self.encoder.output)

        # Decoder part (theta)
        self.decoder = utils.DecoderMLP(encoder_output, ldim, hdim, n_out)

        # Prediction
        self.predict = self.decoder.bern

        # Cost function
        self.kl_div = T.mean(utils.kl_unit_normal(self.encoder.mu_xy, (1.0/self.encoder.lambda_xy)))
        self.xent = T.mean(T.sum(T.nnet.binary_crossentropy(self.decoder.bern, self.input_tensor), axis=1))
        self.cost = self.kl_div + self.xent

        # parameters
        self.params = self.encoder.params + self.decoder.params
        self.updates = adam(self.cost, self.params, self.learning_rate)

        # functions
        self.predict = theano.function(
            inputs=[input_tensor, self.mask],
            outputs=self.predict
        )
        self.cost_fun = theano.function(
            inputs=[input_tensor],
            outputs=[self.cost, self.kl_div, self.xent]
        )
        self.train = theano.function(
            inputs=[input_tensor],
            outputs=[self.cost, self.kl_div, self.xent],
            updates=self.updates
        )
        self.normal_vars = theano.function(
            inputs=[input_tensor],
            outputs=[self.encoder.mu, self.encoder.sigma2]
        )


def main(args):
    # args
    name = args.name
    learning_rate = args.learning_rate
    hdim = args.hdim
    ldim = args.ldim
    epochs = args.epochs
    batch_size = args.batch_size
    timestamp = datetime.datetime.now().isoformat()

    # Save name
    dir_name = '{}_lr{}-hdim{}-ldim{}-bs{}_{}'.format(name,
                                                      learning_rate,
                                                      hdim,
                                                      ldim,
                                                      batch_size,
                                                      timestamp)

    print('Creating save directory.')
    save_path = os.path.join(PATHS['save_dir'], dir_name)
    os.makedirs(save_path)

    # Input matrix and mask boolean
    X = T.matrix('X')
    mask = T.dscalar('mask')

    print('Initializing factored gaussian.')
    # We now define the network
    factor_gaussian = FactoredGaussian(learning_rate,
                                       hdim,
                                       ldim,
                                       X,
                                       mask,
                                       n_in=28*28,
                                       n_out=28*28)
    print('Loading data set.')
    # Load data
    train_set, val_set, test_set = utils.load_data(os.path.join(PATHS['data_raw_dir'], 'mnist.pkl.gz'))
    train_size = train_set.shape[0]
    test_size = test_set.shape[0]
    val_size = val_set.shape[0]

    # Holders
    best_validation_loss = np.inf
    this_validation_loss = 0
    n = 0
    train_cost = 0

    # Plotting arrays
    epoch_arr = []
    train_loss = []
    train_kl_arr = []
    val_loss = []
    val_kl_arr = []

    print('Training model')
    # Train 'epochs' number of times
    for epoch in tqdm(xrange(epochs)):
        # Reset counters
        n = 0
        train_cost = 0
        train_kl = 0
        # For each epoch we shuffle the index
        rand_index = np.random.permutation(test_size)
        for i in xrange(test_size//batch_size):
            # Perform one training step
            n += 1
            temp_cost, temp_kl, temp_xent = auto_nn.train(test_set[rand_index[i * batch_size: (i + 1) * batch_size], :])
            train_cost += temp_cost
            train_kl += temp_kl
        # Get average cost
        train_cost /= n
        train_kl /= n
        # Evaluate every 'eval_frequency'th step
        if epoch % args.eval_frequency == 0:
            n = 0
            val_cost = 0
            val_kl = 0
            for i in xrange(val_size//batch_size):
                n += 1
                temp_cost, temp_kl, temp_xent = auto_nn.train(test_set[rand_index[i * batch_size: (i + 1) * batch_size], :])
                val_cost += temp_cost
                val_kl += temp_kl
            # Average
            val_cost /= n
            val_kl /= n
            print('Epoch: {} | Train cost: {:.3f}, Train KL: {:.3f} | Val cost: {:.3f}, Val KL: {:.3f}'.format(epoch, train_cost, train_kl, val_cost, val_kl))
            epoch_arr.append(epoch)
            train_loss.append(train_cost)
            train_kl_arr.append(train_kl)
            val_loss.append(val_cost)
            val_kl_arr.append(val_kl)
            if val_cost < best_validation_loss:
                best_validation_loss = val_cost
                # Save model
                with open(os.path.join(save_path, 'best_model.pkl'), 'wb') as f:
                    pickle.dump(factor_gaussian.predict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Set matplotlib style
    plt.style.use('seaborn-darkgrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    index = np.random.permutation(test_size)[0: 9*9]
    baseline = test_set[index, :]
    predicted = factor_gaussian.predict(test_set[index, :], 0)
    predicted_masked = factor_gaussian.predict(test_set[index, :], 1)
    # Baseline images
    fig, ax = plt.subplots(9, 9)
    ax = ax.ravel()
    for i, a in enumerate(ax):
        a.imshow(np.reshape(baseline[i, :], [28, 28]), cmap='gray')
        a.axis('off')
    fig.savefig(os.path.join(save_path, 'baseline.png'))
    # Predicted images
    fig, ax = plt.subplots(9, 9)
    ax = ax.ravel()
    for i, a in enumerate(ax):
        a.imshow(np.reshape(predicted[i, :], [28, 28]), cmap='gray')
        a.axis('off')
    fig.savefig(os.path.join(save_path, 'predicted.png'))
    # Baseline masked images
    mask = np.zeros(28 * 28)
    mask[392:] = 1
    mask = mask.reshape(1, -1)
    fig, ax = plt.subplots(9, 9)
    ax = ax.ravel()
    for i, a in enumerate(ax):
        a.imshow(np.reshape(np.ma.masked_array(baseline[i, :], mask=mask), [28, 28]), cmap='gray')
        a.axis('off')
    fig.savefig(os.path.join(save_path, 'baseline_masked.png'))
    # Predicted masked images
    fig, ax = plt.subplots(9, 9)
    ax = ax.ravel()
    for i, a in enumerate(ax):
        a.imshow(np.reshape(predicted_masked[i, :], [28, 28]), cmap='gray')
        a.axis('off')
    fig.savefig(os.path.join(save_path, 'predicted_masked.png'))
    # Loss images
    fig, ax = plt.subplots(1, 1)
    ax.plot(epoch_arr, train_loss, 'r', label='Train loss')
    ax.plot(epoch_arr, train_kl_arr, 'g', label='Train KL')
    ax.plot(epoch_arr, val_loss, 'b', label='Validation loss')
    ax.plot(epoch_arr, val_kl_arr, 'y', label='Validation KL')
    ax.set_xlabel('epochs')
    ax.set_ylim([0, 250])
    ax.set_ylabel('loss')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(save_path, 'Loss.png'))
    plt.clf()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-n",
                        "--name",
                        dest="name",
                        help="Name of the model",
                        default='MNIST_fact_gaussian',
                        type=str)
    parser.add_argument("-l",
                        "--learning_rate",
                        dest="learning_rate",
                        help="Specify the learning rate.",
                        default=0.00001,
                        type=float)
    parser.add_argument("-u",
                        "--hdim",
                        dest="hdim",
                        help="Specify the number of hidden units.",
                        default=100,
                        type=int)
    parser.add_argument("-d",
                        "--ldim",
                        dest="ldim",
                        help="Specify the number of latent dimensions",
                        default=2,
                        type=int)
    parser.add_argument("-e",
                        "--epochs",
                        dest="epochs",
                        help="Specify the number of epochs to run for",
                        default=1000,
                        type=int)
    parser.add_argument("-b",
                        "--batch_size",
                        dest="batch_size",
                        help="Specify the batch size",
                        default=128,
                        type=int)
    parser.add_argument("-f",
                        "--eval_frequency",
                        dest="eval_frequency",
                        help="The evaluation frequency that we validate",
                        default=128,
                        type=int)

    args = parser.parse_args()

    # Run model
    main(args)
