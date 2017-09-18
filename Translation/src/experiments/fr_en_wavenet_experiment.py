# -*- coding: utf-8 -*-

"""
Experiment for WaveNet on translation (en->fr)
"""

import argparse
import os
import sys
import timeit
import time
import cPickle as pickle
from pprint import pprint
import codecs
# so that we can pipe unicode
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

sys.path.append('../models/')
sys.path.append('../data/')
sys.path.append('../external/')
sys.path.append('../tools/')
sys.path.append('../visualization/')
sys.path.append('../../../')

from distributions import GaussianDiagonal, Categorical
from recognition_models import RecWaveNetText as RecognitionModel
from generative_models import GenWaveNetTextWords as GenerativeModel
from SETTINGS import *

import string

import lasagne

from sgvb import SGVBWavenet as SGVB
from run_translate import RunWords

import numpy as np

sys.setrecursionlimit(5000000)
np.set_printoptions(threshold=1000000)


def main(args):
    # directories
    load_param_dir = args.load_param_dir
    if load_param_dir:
        print('Loaded directory used as output directory.')
        load_param_dir = os.path.join(PATHS['save_dir'], load_param_dir)
        out_dir = load_param_dir
    else:
        print('Creating save directory.')
        out_dir = os.path.join(PATHS['save_dir'], 'wavenet_wavenet_translation_fr_en.{}'.format(time.time()))

        # if output directory doesn't exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # solver
    solver = SGVB

    # valid vocabulary
    processed_data_path = os.path.join(PATHS['data_processed_dir'], 'fr_en_2to50_vocab30000_full')
    valid_vocab_path_en = os.path.join(processed_data_path, 'en_valid_vocab.pickle')
    valid_vocab_path_fr = os.path.join(processed_data_path, 'fr_valid_vocab.pickle')

    with open(valid_vocab_path_en) as f:
        valid_vocab_en = pickle.load(f)
    with open(valid_vocab_path_fr) as f:
        valid_vocab_fr = pickle.load(f)

    # size of vocabulary
    vocab_size_en = len(valid_vocab_en)
    vocab_size_fr = len(valid_vocab_fr)

    # counts of the words in the vocabulary
    valid_vocab_counts_path_en = os.path.join(processed_data_path, 'en_valid_vocab_counts.pickle')
    valid_vocab_counts_path_fr = os.path.join(processed_data_path, 'fr_valid_vocab_counts.pickle')

    with open(valid_vocab_counts_path_en) as f:
        valid_vocab_counts_en = pickle.load(f)
    with open(valid_vocab_counts_path_fr) as f:
        valid_vocab_counts_fr = pickle.load(f)

    # restrictions on data
    restrict_min_length = args.restrict_min_len
    restrict_max_length = args.restrict_max_len
    train_prop = args.train_prop

    # dimensions
    d_z = args.d_z
    d_emb = args.d_emb

    # hyperparameters of the rnn's
    hid_depth_gen = args.hid_depth_gen
    use_skip_gen = args.use_skip_gen
    hid_depth_rec = args.hid_depth_rec

    # neural network dictionaries
    gen_nn_kwargs = {
        'dilations': [
            1, 2, # 4, 8, 16,  # 32, 64, 128, 256,
            # 1, 2, 4, 8, 16,  # 32, 64, 128, 256,
        ],
        'dilation_channels': 1,
        'residual_channels': 2,
    }

    rec_nn_kwargs = {
        'dilations': [
            1, 2, 4, 8,  # 16, 32, 64, 128, 256,
            # 1, 2, 4, 8, 16, 32, 64, 128, 256,
            # 1, 2, 4, 8, 16, 32, 64, 128, 256,
        ],
        'dilation_channels': 200,
        'residual_channels': 400,
        'final_depth': 3,
        'final_hid_units': 1000,
        'final_hid_nonlinearity': lasagne.nonlinearities.elu,
    }

    recognition_model = RecognitionModel
    generative_model = GenerativeModel

    # Possibly should pass in an instance instead of a class?
    solver_kwargs = {'generative_model': GenerativeModel,
                     'recognition_model': RecognitionModel,
                     'vocab_size_x': vocab_size_en,
                     'vocab_size_y': vocab_size_fr,
                     'gen_nn_kwargs': gen_nn_kwargs,
                     'rec_nn_kwargs': rec_nn_kwargs,
                     'z_dim': d_z,
                     'embedding_dim': d_emb,
                     'z_dist_gen': GaussianDiagonal,
                     'x_dist_gen': Categorical,
                     'y_dist_gen': Categorical,
                     'z_dist_rec': GaussianDiagonal
                    }

    load_param_dir = args.load_param_dir  # Don't have any saved parameters yet

    train = args.train

    training_iter = args.training_iter
    training_batch_size = args.training_batch_size
    training_num_samples = args.training_num_samples
    warm_up = args.warm_up
    approximate_by_css = args.approximate_by_css
    css_num_samples = args.training_css_num_samples

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

    test = args.test
    test_batch_size = args.test_batch_size
    test_num_samples = args.test_num_samples
    test_sub_sample_size = args.test_sub_sample_size

    translate = args.translate
    translation_source = args.translation_source

    dataset_en = os.path.join(processed_data_path, 'en_word_level_processed.pickle')
    dataset_fr = os.path.join(processed_data_path, 'fr_word_level_processed.pickle')

    # Save all of the variables to stdout so that we know the value of all necessary things
    allvars = dict(out_dir=out_dir,
                   vocab_size_en=vocab_size_en,
                   vocab_size_fr=vocab_size_fr,
                   restrict_min_length=restrict_min_length,
                   restrict_max_length=restrict_max_length,
                   train_prop=train_prop,
                   d_z=d_z,
                   d_emb=d_emb,
                   training_iter=training_iter,
                   training_batch_size=training_batch_size,
                   training_num_samples=training_num_samples,
                   approximate_by_css=approximate_by_css,
                   css_num_samples=css_num_samples,
                   warm_up=warm_up,
                   grad_norm_constraint=grad_norm_constraint,
                   val_freq=val_freq,
                   val_batch_size=val_batch_size,
                   val_num_samples=val_num_samples,
                   test_batch_size=test_batch_size,
                   test_num_samples=test_num_samples,
                   test_sub_sample_size=test_sub_sample_size,
                   gen_nn_kwargs=gen_nn_kwargs,
                   rec_nn_kwargs=rec_nn_kwargs)
    allvars.update(update_kwargs)

    with open(os.path.join(out_dir, 'allvars.save'), 'wb') as f:
        pickle.dump(allvars, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Save them to disk (This serves as a documentation of hyperparameters)
    with open(os.path.join(out_dir, 'args.save'), 'wb') as f:
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)

    # initialise running instance
    run = RunWords(solver=solver,
                   solver_kwargs=solver_kwargs,
                   recognition_model=recognition_model,
                   generative_model=generative_model,
                   valid_vocab_x=valid_vocab_en,
                   valid_vocab_y=valid_vocab_fr,
                   valid_vocab_counts_x=valid_vocab_counts_en,
                   valid_vocab_counts_y=valid_vocab_counts_fr,
                   out_dir=out_dir,
                   dataset_path_x=dataset_en,
                   dataset_path_y=dataset_fr,
                   load_param_dir=load_param_dir,
                   restrict_min_length=restrict_min_length,
                   restrict_max_length=restrict_max_length,
                   train_prop=train_prop)

    if train:
        run.train(num_iter=training_iter,
                  batch_size=training_batch_size,
                  num_samples=1,
                  approximate_by_css=approximate_by_css,
                  css_num_samples=css_num_samples,
                  grad_norm_constraint=grad_norm_constraint,
                  update=update,
                  update_kwargs=update_kwargs,
                  warm_up=warm_up,
                  val_freq=val_freq,
                  val_num_samples=1,#val_num_samples,
                  val_gen_print_samples=5,
                  save_params_every=save_params_every)

        print('='*70)
        print('         COMPLETED TRAINING')
        print('='*70)

    if generate_output_prior or generate_output_posterior:
        run.generate_output(prior=generate_output_prior, posterior=generate_output_posterior, num_outputs=num_outputs)

        print('='*70)
        print('         COMPLETED GENERATING OUTPUT')
        print('='*70)

    if follow_latent_trajectory:
        run.follow_latent_trajectory(num_samples=num_outputs, num_steps=latent_trajectory_steps)

        print('='*70)
        print('         COMPLETED FOLLOW LATENT TRAJECTORY')
        print('='*70)

    if test:
        run.test(test_batch_size, approximate_by_css, css_num_samples, test_num_samples, test_sub_sample_size)

        print('='*70)
        print('         COMPLETED TESTING')
        print('='*70)

    if translate:
        run.test(test_batch_size, approximate_by_css, css_num_samples, test_num_samples, test_sub_sample_size,
                 translate=True, translation_source=translation_source)

        print('='*70)
        print('         COMPLETED TRANSLATION')
        print('='*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("--main_dir",
                        dest="main_dir",
                        help="Specify the main directory.",
                        default=".",
                        type=str)
    parser.add_argument("--min_len",
                        dest="restrict_min_len",
                        help="Specify the minimum length of a sentence",
                        default=2,
                        type=int)
    parser.add_argument("--max_len",
                        dest="restrict_max_len",
                        help="Specify the maximum length of a sentence",
                        default=20,
                        type=int)
    parser.add_argument("--train_prop",
                        dest="train_prop",
                        help="Specify the training proportion.",
                        default=0.9,
                        type=float)
    parser.add_argument("--z_dim",
                        dest="d_z",
                        help="Specify the dimensionality of z.",
                        default=50,
                        type=int)
    parser.add_argument("--d_emb",
                        dest="d_emb",
                        help="Specify the dimensionality of emb.",
                        default=300,
                        type=int)
    parser.add_argument("--hid_depth_gen",
                        dest="hid_depth_gen",
                        help="Specify the depth of the generating model",
                        default=3,
                        type=int)
    parser.add_argument("--use_skip_gen",
                        dest="use_skip_gen",
                        help="If we are to use skip-connection",
                        default=0,
                        type=int)
    parser.add_argument("--hid_depth_rec",
                        dest="hid_depth_rec",
                        help="Specify the depth of the recognition model",
                        default=3,
                        type=int)
    parser.add_argument("--load_param_dir",
                        dest="load_param_dir",
                        help="Specify if the directory of the saved parameters",
                        default=None,
                        type=str)
    parser.add_argument("--train",
                        dest="train",
                        help="Specify if we want to train the model",
                        default=1,
                        type=int)
    parser.add_argument("--training_iter",
                        dest="training_iter",
                        help="Specify how many iterations we want to train the model",
                        default=100000,
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
                        default=10000,
                        type=int)
    parser.add_argument("--approximate_by_css",
                        dest="approximate_by_css",
                        help=" ",
                        default=False,
                        type=bool)
    parser.add_argument("--training_css_num_samples",
                        dest="training_css_num_samples",
                        help=" ",
                        default=100,
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
                        default=500,
                        type=int)
    parser.add_argument("--val_batch_size",
                        dest="val_batch_size",
                        help="Specify the validation batch size",
                        default=200,
                        type=int)
    parser.add_argument("--val_num_samples",
                        dest="val_num_samples",
                        help="Specify the number of validation samples when calculating ELBO",
                        default=1,
                        type=int)
    parser.add_argument("--save_params_every",
                        dest="save_params_every",
                        help="How often to save parameters while training",
                        default=2000,
                        type=int)
    parser.add_argument("--generate_output_prior",
                        dest="generate_output_prior",
                        help="If we want to print the generated sequence from prior",
                        default=1,
                        type=int)
    parser.add_argument("--generate_output_posterior",
                        dest="generate_output_posterior",
                        help="If we want to print the generated sequence from posterior",
                        default=1,
                        type=int)
    parser.add_argument("--follow_latent_trajectory",
                        dest="follow_latent_trajectory",
                        help="If we want to generate paths in the latent space (homotopies)",
                        default=0,
                        type=int)
    parser.add_argument("--latent_trajectory_steps",
                        dest="latent_trajectory_steps",
                        help="Number of random trajectories (homotopies) to generate",
                        default=10,
                        type=int)
    parser.add_argument("--num_outputs",
                        dest="num_outputs",
                        help="Number of outputs to generate",
                        default=40,
                        type=int)
    parser.add_argument("--test",
                        dest="test",
                        help="If we are to do test phase as well",
                        default=1,
                        type=int)
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
                        default=1,
                        type=int)
    parser.add_argument("--translate",
                        dest="translate",
                        help="If we are to do translation phase as well",
                        default=0,
                        type=int)
    parser.add_argument("--translation_source",
                        dest="translation_source",
                        help="The language that we want to translate from",
                        default="x",
                        type=str)

    args = parser.parse_args()
    
    # Run model
    main(args)
