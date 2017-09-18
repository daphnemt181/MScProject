# -*- coding: utf-8 -*-

"""
Experiment for AUTR on translation (en->fr)
"""

import argparse
import os
import sys
import timeit
import time
from pprint import pprint

sys.path.append('./src/models/')
sys.path.append('./src/data/')
sys.path.append('./src/external/')
sys.path.append('./src/tools/')
sys.path.append('./src/visualization/')
sys.path.append('.')

import theano
import theano.tensor as T
import lasagne

theano.config.floatX = 'float32'

from distributions import GaussianDiagonal, Categorical
from recognition_models import FacGaussRecModel
from generative_models import GenAUTR
from SETTINGS import *

from sgvb import SGVBAUTR as SGVB
from run_translate import RunTranslation

sys.setrecursionlimit(5000000)

def main(args):
    # directories
    main_dir = args.main_dir

    print('Creating save directory.')
    out_dir = os.path.join('..', 'output', 'lr{}hdim{}zdim{}-{}'.format(args.learning_rate,
                                                                        args.d_hid,
                                                                        args.d_z,
                                                                        time.time()))

    # if output directory doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # solver
    solver = SGVB

    # valid vocabulary
    # Include all English and French lowercase characters (already imported from SETTINGS)
    # valid_vocab_en = valid_vocab_en
    # valid_vocab_fr = valid_vocab_fr

    # size of vocabulary
    vocab_size_en = len(valid_vocab_en)
    vocab_size_fr = len(valid_vocab_fr)

    # restrictions on data
    restrict_max_length = args.restrict_max_len
    train_prop = args.train_prop

    # dimensions
    d_z = args.d_z
    d_hid = args.d_hid
    num_time_steps = args.num_time_steps

    # hyperparameters of the rnn's
    hid_depth_gen = args.hid_depth_gen
    use_skip_gen = args.use_skip_gen
    hid_depth_rec = args.hid_depth_rec

    # neural network dictionaries
    rec_nn_kwargs = {
        'nn_depth': hid_depth_rec,
        'nonlinearity': lasagne.nonlinearities.tanh,
        'dropout_p': 0.0}

    gen_nn_kwargs = {
        'rnn_depth': 5,
        'use_skip': True,
        'hid_dim': d_hid,
        'bidirectional': True}

    recognition_model = FacGaussRecModel
    generative_model = GenAUTR

    # Possibly should pass in an instance instead of a class?
    solver_kwargs = {'generative_model': generative_model,
                     'recognition_model': recognition_model,
                     'vocab_size_x': vocab_size_en,
                     'vocab_size_y': vocab_size_fr,
                     'num_time_steps': args.num_time_steps,
                     'gen_nn_kwargs': gen_nn_kwargs,
                     'rec_nn_kwargs': rec_nn_kwargs,
                     'z_dim': d_z,
                     'z_dist_gen': GaussianDiagonal,
                     'x_dist_gen': Categorical,
                     'y_dist_gen': Categorical,
                     'z_dist_rec': GaussianDiagonal}

    load_param_dir = args.load_param_dir  # Don't have any saved parameters yet

    train = args.train

    training_iterations = args.training_iterations
    training_batch_size = args.training_batch_size
    training_num_samples = args.training_num_samples
    warm_up = args.warm_up

    grad_norm_constraint = args.grad_norm_constraint
    update = lasagne.updates.adam
    update_kwargs = {'learning_rate': args.learning_rate}

    val_freq = args.val_freq
    val_batch_size = args.val_batch_size
    val_num_samples = args.val_num_samples

    save_params_every = args.save_params_every

    generate_output_prior = args.generate_output_prior
    generate_output_posterior = args.generate_output_posterior

    follow_latent_trajectory = args.follow_latent_trajectory
    latent_trajectory_steps = args.latent_trajectory_steps

    num_outputs = args.num_outputs

    # test = args.test
    # test_batch_size = args.test_batch_size
    # test_num_samples = args.test_num_samples
    # test_sub_sample_size = args.test_sub_sample_size

    processed_data_path = os.path.join('.', 'data', 'processed')

    dataset_en = os.path.join(processed_data_path, 'en_processed_toy.pickle')
    dataset_fr = os.path.join(processed_data_path, 'fr_processed_toy.pickle')

    # initialise running instance
    run = RunTranslation(solver=SGVB,
                         solver_kwargs=solver_kwargs,
                         recognition_model=recognition_model,
                         generative_model=generative_model,
                         valid_vocab_x=valid_vocab_en,
                         valid_vocab_y=valid_vocab_fr,
                         out_dir=out_dir,
                         dataset_path_x=dataset_en,
                         dataset_path_y=dataset_fr,
                         load_param_dir=load_param_dir,
                         restrict_max_length=restrict_max_length,
                         train_prop=train_prop)

    # PLAYGROUND
    # input tensors
    x_t = T.imatrix('x')
    y_t = T.imatrix('y')

    def output_fn(x, y):
        x_ohe = run.sgvb.one_hot_encode(x, run.sgvb.one_hot_encoder_x)
        y_ohe = run.sgvb.one_hot_encode(y, run.sgvb.one_hot_encoder_y)

        z = run.sgvb.recognition_model.get_samples(x_ohe, y_ohe, num_samples=1)

        log_q_z = run.sgvb.recognition_model.log_q_z(z, x_ohe, y_ohe)  # (S*N)
        log_p_z = run.sgvb.generative_model_x.log_p_z(z)  # (S*N)
        log_p_x = run.sgvb.generative_model_x.log_p_x(x_ohe, z)  # (S*N)
        log_p_y = run.sgvb.generative_model_y.log_p_x(y_ohe, z)  # (S*N)

        # symbolic output functions for x and y
        posterior_output_x = run.sgvb.generative_model_x.generate_output_posterior_fn(x, y, z, num_samples_per_sample=1, only_final=True)
        posterior_output_y = run.sgvb.generative_model_y.generate_output_posterior_fn(y, x, z, num_samples_per_sample=1, only_final=True)


        output = theano.function([x, y],
                                 [x_ohe,
                                  y_ohe,
                                  z,
                                  log_q_z,
                                  log_p_z,
                                  log_p_x,
                                  log_p_y],
                                 allow_input_downcast=True)

        return output, posterior_output_x, posterior_output_y

    output, post_x, post_y = output_fn(x_t, y_t)

    print(post_x(run.x_test[0:3], run.y_test[0:3]))
    print(post_y(run.y_test[0:3], run.x_test[0:3]))

    x_o, y_o, z_o, lq_z, lp_z, lp_x, lp_y = output(run.x_test[0:3], run.y_test[0:3])
    print('shapes')
    print('x_o: {}, y_o: {}, z_o: {}, lq_z: {}, lp_z: {}, lp_x: {}, lp_y: {}'.format(x_o.shape, y_o.shape, z_o.shape, lq_z.shape, lp_z.shape, lp_z.shape, lp_y.shape))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("--main_dir",
                        dest="main_dir",
                        help="Specify the main directory.",
                        default=".",
                        type=str)
    # parser.add_argument("--out_dir",
    #                     dest="out_dir",
    #                     help="Specify the output directory.",
    #                     default=".",
    #                     type=str)
    parser.add_argument("--max_len",
                        dest="restrict_max_len",
                        help="Specify the maximum length of a sentence",
                        default=50,
                        type=int)
    parser.add_argument("--train_prop",
                        dest="train_prop",
                        help="Specify the training proportion.",
                        default=0.9,
                        type=float)
    parser.add_argument("--z_dim",
                        dest="d_z",
                        help="Specify the dimensionality of z.",
                        default=2000,
                        type=int)
    parser.add_argument("--hid_dim",
                        dest="d_hid",
                        help="Specify the dimensionality of h.",
                        default=500,
                        type=int)
    parser.add_argument("--num_time_steps",
                        dest="num_time_steps",
                        help="Specify the number of time steps to use in RNN",
                        default=30,
                        type=int)
    parser.add_argument("--hid_depth_gen",
                        dest="hid_depth_gen",
                        help="Specify the depth of the generating model",
                        default=3,
                        type=int)
    parser.add_argument("--use_skip_gen",
                        dest="use_skip_gen",
                        help="If we are to use skip-connection",
                        default=False,
                        type=bool)
    parser.add_argument("--hid_depth_rec",
                        dest="hid_depth_rec",
                        help="Specify the depth of the recognition model",
                        default=3,
                        type=int)
    parser.add_argument("--pre_trained",
                        dest="pre_trained",
                        help="Specify if the model is pretrained",
                        default=False,
                        type=bool)
    parser.add_argument("--load_param_dir",
                        dest="load_param_dir",
                        help="Specify if the directory of the saved parameters",
                        default=None,
                        type=str)
    parser.add_argument("--train",
                        dest="train",
                        help="Specify if we want to train the model",
                        default=True,
                        type=bool)
    parser.add_argument("--training_iterations",
                        dest="training_iterations",
                        help="Specify how many iterations we want to train the model",
                        default=500000,
                        type=int)
    parser.add_argument("--training_batch_size",
                        dest="training_batch_size",
                        help="Specify the batch size for training",
                        default=200,
                        type=int)
    parser.add_argument("--training_num_samples",
                        dest="training_num_samples",
                        help="Specify the number of samples (num. of z's) to sample in SGVB",
                        default=1,
                        type=int)
    parser.add_argument("--warm_up",
                        dest="warm_up",
                        help="Specify the warm up period, the annealing from 0 to 1 for KL term",
                        default=100000,
                        type=int)
    parser.add_argument("--grad_norm_constraint",
                        dest="grad_norm_constraint",
                        help="Specify any gradient constraint while updating",
                        default=None)
    parser.add_argument("--learning_rate",
                        dest="learning_rate",
                        help="Specify the learning rate",
                        default=0.0001,
                        type=float)
    parser.add_argument("--val_freq",
                        dest="val_freq",
                        help="Specify the validation frequency to print",
                        default=1000,
                        type=int)
    parser.add_argument("--val_batch_size",
                        dest="val_batch_size",
                        help="Specify the validation batch size",
                        default=5,
                        type=int)
    parser.add_argument("--val_num_samples",
                        dest="val_num_samples",
                        help="Specify the number of validation samples when calculating ELBO",
                        default=5,
                        type=int)
    parser.add_argument("--save_params_every",
                        dest="save_params_every",
                        help="How often to save parameters while training",
                        default=50000,
                        type=int)
    parser.add_argument("--generate_output_prior",
                        dest="generate_output_prior",
                        help="If we want to print the generated sequence from prior",
                        default=True,
                        type=bool)
    parser.add_argument("--generate_output_posterior",
                        dest="generate_output_posterior",
                        help="If we want to print the generated sequence from posterior",
                        default=True,
                        type=bool)
    parser.add_argument("--find_best_matches",
                        dest="find_best_matches",
                        help="If we want to find the best matches to a sentence",
                        default=True,
                        type=bool)
    parser.add_argument("--num_best_matches",
                        dest="num_best_matches",
                        help="Number of best matches to find (samples)",
                        default=20,
                        type=int)
    parser.add_argument("--best_matches_batch_size",
                        dest="best_matches_batch_size",
                        help="Batch size of number of best matches",
                        default=100,
                        type=int)
    parser.add_argument("--follow_latent_trajectory",
                        dest="follow_latent_trajectory",
                        help="If we want to generate paths in the latent space (homotopies)",
                        default=True,
                        type=bool)
    parser.add_argument("--latent_trajectory_steps",
                        dest="latent_trajectory_steps",
                        help="Number of random trajectories (homotopies) to generate",
                        default=10,
                        type=int)
    parser.add_argument("--num_outputs",
                        dest="num_outputs",
                        help="Number of outputs to generate",
                        default=100,
                        type=int)
    parser.add_argument("--test",
                        dest="test",
                        help="If we are to do test phase as well",
                        default=True,
                        type=bool)
    parser.add_argument("--test_batch_size",
                        dest="test_batch_size",
                        help="Specify batch size while testing",
                        default=500,
                        type=int)
    parser.add_argument("--test_num_samples",
                        dest="test_num_samples",
                        help="Specify number of samples while testing",
                        default=100,
                        type=int)
    parser.add_argument("--test_sub_sample_size",
                        dest="test_sub_sample_size",
                        help="Specify number of sub-samples to use while testing",
                        default=100,
                        type=int)

    args = parser.parse_args()

    # Run model
    main(args)
