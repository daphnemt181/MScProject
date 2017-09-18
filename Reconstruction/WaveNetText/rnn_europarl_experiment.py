from model.distributions import GaussianDiagonal, Categorical
from model.generative_models import GenWaveNetTextWords as GenerativeModel
from model.recognition_models import RecRNN as RecognitionModel

import json
import os
import sys
import cPickle as pickle
import lasagne
import time
import datetime
import sys
import codecs
# so that we can pipe unicode
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

from trainers.sgvb import SGVBWords as SGVB
from run import RunWords as Run

from pprint import *

sys.path.append('../../')

from SETTINGS import *

import numpy as np
np.set_printoptions(threshold=1000000)

sys.setrecursionlimit(5000000)
# need to check this one

lang = 'en'

main_dir = '.'

data_folder = os.path.join(PATHS['data_processed_dir'], 'fr_en_2to50_vocab30000_full')
dataset = os.path.join(data_folder, '{}_word_level_processed.pickle'.format(lang))

with open(os.path.join(data_folder, '{}_valid_vocab.pickle'.format(lang)), 'r') as f:
    valid_vocab = pickle.load(f)

with open(os.path.join(data_folder, '{}_valid_vocab_counts.pickle'.format(lang)), 'r') as f:
    valid_vocab_counts = pickle.load(f)

solver = SGVB

iwae = False


vocab_size = len(valid_vocab)
restrict_min_length = 2
restrict_max_length = 20
train_prop = 0.9

d_z = 50
d_emb = 300

gen_nn_kwargs = {
    'dilations': [
        1, 2, 4, 8,  # 16,
        # 1, 2, 4, 8, 16,
        # 1, 2, 4, 8, 16,
    ],
    'dilation_channels': 2,
    'residual_channels': 4,
}

rec_nn_kwargs = {
    'rnn_hid_dim': 1000,
    'final_depth': 3,
    'final_hid_units': 1000,
    'final_hid_nonlinearity': lasagne.nonlinearities.elu,
}

solver_kwargs = {'generative_model': GenerativeModel,
                 'recognition_model': RecognitionModel,
                 'vocab_size': vocab_size,
                 'gen_nn_kwargs': gen_nn_kwargs,
                 'rec_nn_kwargs': rec_nn_kwargs,
                 'z_dim': d_z,
                 'embedding_dim': d_emb,
                 'dist_z_gen': GaussianDiagonal,
                 'dist_x_gen': Categorical,
                 'dist_z_rec': GaussianDiagonal,
                 'iwae': iwae,
                 }

pre_trained = False
load_param_dir = ''

if pre_trained:
    print('Pre-trained, setting directory to: {}'.format(load_param_dir))
    out_dir = os.path.join(PATHS['save_dir'], load_param_dir)
    load_param_dir = out_dir
else:
    timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    print('Creating save directory.')
    out_dir = os.path.join(PATHS['save_dir'], 'rnn_reconstruction_output_lang_{}_{}'.format(lang, timestamp))
    # if output directory doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


train = True

training_iterations = 100000
training_batch_size = 200
training_num_samples = 1
approximate_by_css = False
training_css_num_samples = 100
warm_up = 10000

grad_norm_constraint = None
update = lasagne.updates.adam
update_kwargs = {'learning_rate': 0.0001}

val_freq = 1000
val_batch_size = 100
val_num_samples = 1

save_params_every = 10000


generate_output_prior = True
generate_output_posterior = True

impute_missing_chars = False
drop_rate = 0.2
missing_chars_iterations = 500

find_best_matches = False
num_best_matches = 20
best_matches_batch_size = 100

follow_latent_trajectory = False
latent_trajectory_steps = 10

num_outputs = 100

test = True
test_batch_size = 200
test_num_samples = 1
test_sub_sample_size = None

if __name__ == '__main__':
    # Save all of the variables to stdout so that we know the value of all necessary things
    allvars = dict(out_dir=out_dir,
                   iwae=iwae,
                   vocab_size=vocab_size,
                   restrict_min_length=restrict_min_length,
                   restrict_max_length=restrict_max_length,
                   train_prop=train_prop,
                   d_z=d_z,
                   d_emb=d_emb,
                   training_iterations=training_iterations,
                   training_batch_size=training_batch_size,
                   training_num_samples=training_num_samples,
                   approximate_by_css=approximate_by_css,
                   training_css_num_samples=training_css_num_samples,
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

    print('Creating Run instance')
    run = Run(dataset=dataset, valid_vocab=valid_vocab, valid_vocab_counts=valid_vocab_counts, solver=solver,
              solver_kwargs=solver_kwargs, main_dir=main_dir, out_dir=out_dir, pre_trained=pre_trained,
              load_param_dir=load_param_dir, restrict_min_length=restrict_min_length,
              restrict_max_length=restrict_max_length, train_prop=train_prop)

    if train:
        run.train(n_iter=training_iterations, batch_size=training_batch_size, num_samples=training_num_samples,
                  approximate_by_css=approximate_by_css, css_num_samples=training_css_num_samples,
                  grad_norm_constraint=grad_norm_constraint, update=update, update_kwargs=update_kwargs,
                  val_freq=val_freq, val_batch_size=val_batch_size, val_num_samples=val_num_samples, warm_up=warm_up,
                  save_params_every=save_params_every)

    if generate_output_prior or generate_output_posterior:
        run.generate_output(prior=generate_output_prior, posterior=generate_output_posterior, num_outputs=num_outputs)

    if impute_missing_chars:
        run.impute_missing_chars(num_outputs=num_outputs, drop_rate=drop_rate, num_iterations=missing_chars_iterations)

    if find_best_matches:
        run.find_best_matches(num_outputs=num_outputs, num_matches=num_best_matches, batch_size=best_matches_batch_size)

    if follow_latent_trajectory:
        run.follow_latent_trajectory(num_samples=num_outputs, num_steps=latent_trajectory_steps)

    if test:
        run.test(test_batch_size, test_num_samples, test_sub_sample_size)
