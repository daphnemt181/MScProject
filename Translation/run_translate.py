# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict
import os
import sys
import cPickle as pickle
import codecs
import time
import numpy as np
import theano
from lasagne.updates import adam
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize

sys.path.append('./src/data')

theano.config.floatX = 'float32'

from data_processing import chunker
from SETTINGS import *

class RunTranslation(object):
    def __init__(self,
                 solver,
                 solver_kwargs,
                 recognition_model,
                 generative_model,
                 valid_vocab_x,
                 valid_vocab_y,
                 out_dir,
                 dataset_path_x,
                 dataset_path_y,
                 load_param_dir=None,
                 restrict_max_length=None,
                 train_prop=0.95):
        """
        :param solver: solver class that handles sgvb training and updating
        :param solver_kwargs: kwargs for solver
        :param recognition_model: instance of the recognition model class
        :param generative_model: instance of the generative model class
        :param valid_vocab_x: valid vocabulary for x
        :param valid_vocab_y: valid vocabulary for y
        :param out_dir: path to out directory
        :param dataset_path_x: path to dataset of x
        :param dataset_path_y: path to dataset of y
        :param load_param_dir: path to directory of saved variables. If None, train from start
        :param restricted_max_length: restrict the max lengths of the sentences
        :param train_prop: how much of the original data should be split into training/test set
        """
        # set all attributes
        self.solver = solver
        # solver kwargs are the following
        # generative_model
        # recognition_model
        # max_len_x
        # max_len_y
        # vocab_size_x
        # vocab_size_y
        # num_time_steps
        # gen_nn_kwargs
        # rec_nn_kwargs
        # z_dim
        # z_dist_gen
        # x_dist_gen
        # y_dist_gen
        # z_dist_rec
        self.solver_kwargs = solver_kwargs
        self.recognition_model = recognition_model
        self.generative_model = generative_model
        self.valid_vocab_x = valid_vocab_x
        self.valid_vocab_y = valid_vocab_y
        self.out_dir = out_dir
        self.dataset_path_x = dataset_path_x
        self.dataset_path_y = dataset_path_y
        self.load_param_dir = load_param_dir
        self.restrict_max_length = restrict_max_length
        self.train_prop = train_prop

        # data sets
        self.x_train, self.x_test, self.y_train, self.y_test, self.L_x_train, self.L_x_test, self.L_y_train, self.L_y_test = self.load_data(train_prop, restrict_max_length)

        print('All data sets loaded')
        print('#data points (train): {}, #data points (Test): {}'.format(len(self.L_x_train), len(self.L_x_test)))

        # Number of training and test examples
        # Might need to use validation dataset as well
        self.train_size = len(self.L_x_train)
        self.test_size = len(self.L_x_test)

        # # max_length from the actual data set and instantiate the solver
        self.max_length_x = np.concatenate((self.x_train, self.x_test), axis=0).shape[1]
        self.max_length_y = np.concatenate((self.y_train, self.y_test), axis=0).shape[1]
        # self.sgvb = solver(max_length=self.max_length, **self.solver_kwargs)

        print('Maximum length of sentence (x, y): ({}, {})'.format(self.max_length_x, self.max_length_x))

        # initialise sgvb solver (Check how it is done now)
        self.sgvb = self.solver(max_len_x=self.max_length_x,
                                max_len_y=self.max_length_y,
                                **self.solver_kwargs)

        # if pretrained, load saved parameters of the model and set
        # the parameters of the recognition/generative models
        if load_param_dir is not None:
            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.sgvb.recognition_model.set_param_values(pickle.load(f))
            with open(os.path.join(self.load_param_dir, 'gen_params_x.save'), 'rb') as f:
                self.sgvb.generative_model_x.set_param_values(pickle.load(f))
            with open(os.path.join(self.load_param_dir, 'gen_params_y.save'), 'rb') as f:
                self.sgvb.generative_model_y.set_param_values(pickle.load(f))
            with open(os.path.join(self.load_param_dir, 'all_embeddings_x.save'), 'rb') as f:
                self.sgvb.all_embeddings_x.set_value(pickle.load(f))
            with open(os.path.join(self.load_param_dir, 'all_embeddings_y.save'), 'rb') as f:
                self.sgvb.all_embeddings_y.set_value(pickle.load(f))
            print('Parameters loaded and set.')

    def load_data(self, train_prop, restrict_max_length):
        """Load data set to use for training and testing

        :param train_prop: (float) float in [0, 1] indicating proportion of train/test split
        :param restrict_max_length: (int) upper restriction on the max lengths of sentences"""
        # We load the lists from the pickle files
        # datasets is of the form of list of lists,
        # each list consist of numbers from index of the
        # vocabulary. So N * max(L) list of lists of int.
        with open(self.dataset_path_x) as f:
            dataset_x = pickle.load(f)
        with open(self.dataset_path_y) as f:
            dataset_y = pickle.load(f)

        # words are interpreted abstractly (can be chars or words)
        words_x = []
        words_y = []

        # iterate over sentences
        if restrict_max_length is not None:
            for sent_x, sent_y in zip(dataset_x, dataset_y):
                # filtnner out the sentences that are longer than restrict_max_length
                if len(sent_x) <= restrict_max_length and len(sent_y) <= restrict_max_length:
                    words_x.append(sent_x)
                    words_y.append(sent_y)
        else:
            words_x = dataset_x
            words_y = dataset_y

        # lengths of all of the words in source and target dataset
        L_x = np.array([len(sent_x) for sent_x in words_x])
        L_y = np.array([len(sent_y) for sent_y in words_y])

        # Numpy broadcasting to create a mask N * max(L)
        # the mask is such that it is True when the index
        # has a valid character, False when the original sentence
        # is done (When we have gone into the padding)
        pad_x = L_x[:, None] > np.arange(max(L_x))
        pad_y = L_y[:, None] > np.arange(max(L_y))

        # padd the sentences with zeros after they have ended
        words_to_return_x = np.full(pad_x.shape, 0, dtype='int')
        words_to_return_x[pad_x] = np.concatenate(words_x)
        words_to_return_y = np.full(pad_y.shape, 0, dtype='int')
        words_to_return_y[pad_y] = np.concatenate(words_y)

        # split the train/test data
        split = int(len(words_x) * train_prop)

        # return objects. Train and test set for x and y and lengths
        x_train = words_to_return_x[0: split]
        x_test = words_to_return_x[split:]
        y_train = words_to_return_y[0: split]
        y_test = words_to_return_y[split:]
        L_x_train = L_x[0: split]
        L_x_test = L_x[split:]
        L_y_train = L_y[0: split]
        L_y_test = L_y[split:]

        return x_train, x_test, y_train, y_test, L_x_train, L_x_test, L_y_train, L_y_test

    def call_generate_output_prior(self, generate_output_prior_x, generate_output_prior_y):
        """Call the generate_output_prior function and collect the output

        Works with x or y depending on which function you pass to it.

        :param generate_output_prior: (function) generates the output from AUTR

        :return out: (OrderedDict) dictionary of all the relevant quantities"""
        z, trans_probs_x, canvas_gate_sums_x, viterbi_x, probs_viterbi_x, sampled_x = generate_output_prior_x()
        z, trans_probs_y, canvas_gate_sums_y, viterbi_y, probs_viterbi_y, sampled_y = generate_output_prior_y()

        out = OrderedDict()
        out['generated_z_prior'] = z
        out['generated_trans_probs_prior_x'] = trans_probs_x
        out['generated_canvas_gate_sums_prior_x'] = canvas_gate_sums_x
        out['generated_viterbi_prior_x'] = viterbi_x
        out['generated_probs_viterbi_prior_x'] = probs_viterbi_x
        out['generated_sampled_prior_x'] = sampled_x
        out['generated_trans_probs_prior_y'] = trans_probs_y
        out['generated_canvas_gate_sums_prior_y'] = canvas_gate_sums_y
        out['generated_viterbi_prior_y'] = viterbi_y
        out['generated_probs_viterbi_prior_y'] = probs_viterbi_y
        out['generated_sampled_prior_y'] = sampled_y

        return out

    def print_output_prior(self, output_prior, language_x, language_y):
        """Print the output from the prior on z

        Print the output as generated form the call_generate_output_prior
        function. Agnostic with respect to x or y, but have to be consistent
        in terms of vocabulary.

        :param output_prior: (OrderedDict) dictionary of the quantities from prior output
        :param valid_vocab: (string) the valid vocabulary for the generated output
        :param language: (str) language string, to show what language is being output"""
        # The sentences generated by viterbi (most probably sentences)
        viterbi_x = output_prior['generated_viterbi_prior_x']
        viterbi_y = output_prior['generated_viterbi_prior_y']
        char_index_gen_x = self.valid_vocab_x
        char_index_gen_y = self.valid_vocab_y

        print('='*10)

        # print generated sequences from  viterbi
        for n in range(viterbi_x.shape[0]):
            # TODO: Check that this works with our one-hot mapping, and in general
            print('gen ' + language_x + ' viterbi: ' + ''.join([char_index_gen_x[int(i)] for i in viterbi_x[n]]))
            print('gen ' + language_y + ' viterbi: ' + ''.join([char_index_gen_y[int(i)] for i in viterbi_y[n]]))
            print('-'*10)

        print('='*10)

    # TODO: Check that this works now that we have two languages
    def call_generate_output_posterior(self, generate_output_posterior_x, generate_output_posterior_y, x, y):
        """Call the generate_output_posterior function and collect the output

        Works with x or y depending on which function you pass to it.

        :param generate_output_prior: (function) generates the output from AUTR
        :param x: (np.array) batch input from language x
        :param y: (np.array) batch input from language y

        :return out: (OrderedDict) dictionary of all the relevant quantities"""
        # NOTE: z_x == z_y since we only take the means and don't sample
        # which means that the output is deterministic
        z_x, trans_probs_x, canvas_gate_sums_x, viterbi_x, probs_viterbi_x, sampled_x = generate_output_posterior_x(x, y)
        z_y, trans_probs_y, canvas_gate_sums_y, viterbi_y, probs_viterbi_y, sampled_y = generate_output_posterior_y(y, x)

        out = OrderedDict()

        out['generated_z_posterior'] = z_x
        out['true_posterior_x'] = x
        out['generated_trans_probs_posterior_x'] = trans_probs_x
        out['generated_canvas_gate_sums_posterior_x'] = canvas_gate_sums_x
        out['generated_viterbi_posterior_x'] = viterbi_x
        out['generated_probs_viterbi_posterior_x'] = probs_viterbi_x
        out['generated_sampled_posterior_x'] = sampled_x
        out['true_posterior_y'] = y
        out['generated_trans_probs_posterior_y'] = trans_probs_y
        out['generated_canvas_gate_sums_posterior_y'] = canvas_gate_sums_y
        out['generated_viterbi_posterior_y'] = viterbi_y
        out['generated_probs_viterbi_posterior_y'] = probs_viterbi_y
        out['generated_sampled_posterior_y'] = sampled_y

        return out

    # make more sense to show input sentences for both x and y and then show the
    # sentences generated through q(z|x, y)
    def print_output_posterior(self, output_posterior, language_x, language_y):
        """Print the output from the posterior on z

        Print the output as generated form the call_generate_output_posterior
        function. Agnostic with respect to x or y, but have to be consistent
        in terms of vocabulary.

        :param output_posterior: (OrderedDict) dictionary of the quantities from prior output
        :param valid_vocab: (string) the valid vocabulary for the generated output
        :param language: (str) language string, to show what language is being output"""
        x = output_posterior['true_posterior_x']
        viterbi_x = output_posterior['generated_viterbi_posterior_x']
        y = output_posterior['true_posterior_y']
        viterbi_y = output_posterior['generated_viterbi_posterior_y']

        char_index_gen_x = self.valid_vocab_x
        char_index_gen_y = self.valid_vocab_y

        print('='*10)

        # print generated posterior sequences from viterbi
        for n in range(y.shape[0]):
            print('        true ' + language_x + ': ' + ''.join([char_index_gen_x[i] for i in x[n]]).strip())
            print(' gen ' + language_x + ' viterbi: ' + ''.join([char_index_gen_x[int(i)] for i in viterbi_x[n]]))
            print('        true ' + language_y + ': ' + ''.join([char_index_gen_y[i] for i in y[n]]).strip())
            print(' gen ' + language_y + ' viterbi: ' + ''.join([char_index_gen_y[int(i)] for i in viterbi_y[n]]))
            print('-'*10)

        print('='*10)

    def train(self,
              num_iter,
              batch_size,
              num_samples,
              grad_norm_constraint,
              update,
              update_kwargs,
              warm_up,
              val_freq,
              val_num_samples,
              val_gen_print_samples,
              save_params_every,
              print_every):
        """Train the model

        :param num_iter: number of iterations to run for
        :param batch_size: batch_size
        :param num_samples: number of samples for elbo
        :param grad_norm_constraint: gradient constraints
        :param update: update function
        :param update_kwargs: kwargs for update function
        :param warm_up: if warm up
        :param val_freq: validation frequency
        :param val_num_samples: number of validation samples
        :param val_gen_print_samples: how many generated validation sentences to print
        :param save_params_every: save parameters every _'th iteration
        :param print_every: print to screen every _'th iteration"""
        if self.load_param_dir is not None:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = pickle.load(f)
        else:
            saved_update = None

        # Create optimiser which is the optimiser
        # function and the updates that comes from it
        optimiser, updates = self.sgvb.optimiser(num_samples=num_samples,
                                                 grad_norm_constraint=grad_norm_constraint,
                                                 update=update,
                                                 update_kwargs=update_kwargs,
                                                 saved_update=saved_update)

        # the actual elbo function that we are trying to optimise
        elbo_fn = self.sgvb.elbo_fn(val_num_samples)

        # the symbolic functions which generate output from the latent
        generate_output_prior_x, generate_output_prior_y = self.sgvb.generate_output_prior_fn(val_gen_print_samples)
        generate_output_posterior_x, generate_output_posterior_y = self.sgvb.generate_output_posterior_fn(val_gen_print_samples)

        # Create lists for holding information
        elbo_train_list = []
        elbo_val_list = []
        KL_train_list = []
        KL_val_list = []
        output_prior_list = []
        output_posterior_train_list = []
        output_posterior_test_list = []
        iter_list = []

        # number of batches per epoch
        num_batches_per_epoch = self.train_size/batch_size
        print('number of batches per epoch {}'.format(num_batches_per_epoch))

        # keep track of how many we have gone over in this epoch
        epoch_batch_counter = 0
        # shuffle initial index
        shuffled_index = np.random.permutation(self.train_size)
        # get starting time
        tic = time.clock()

        # For training we train over epochs
        for i in range(num_iter):
            # if we are finished with iteration, reshuffle
            if epoch_batch_counter + 1 >= num_batches_per_epoch:
                shuffled_index = np.random.permutation(self.train_size)
                epoch_batch_counter = 0

            # get batches
            x_batch = self.x_train[shuffled_index[epoch_batch_counter * batch_size: (epoch_batch_counter + 1) * batch_size]]
            y_batch = self.y_train[shuffled_index[epoch_batch_counter * batch_size: (epoch_batch_counter + 1) * batch_size]]

            # annealing constant (beta is a linear up to warm_up, where it is constant 1)
            if warm_up is not None:
                beta = max(1.0, float(i) / warm_up)
            else:
                beta = 1.0

            # run the training step and get the approximate loss function for
            # this step. KL tells us how much of x, y is encoded in z. The
            # bigger KL is the more we are using the latent space.
            elbo, kl = optimiser(x_batch, y_batch, beta)

            if print_every is not None and i % print_every == 0:
                # stop clock
                toc = time.clock()
                # we will sample 'sample' random indices from test and train and get ELBO and KL
                sample = 10
                random_index_test = np.random.choice(self.x_test.shape[0], size=(sample), replace=False)
                random_index_train = np.random.choice(self.x_train.shape[0], size=(sample), replace=False)
                train_elbo, train_kl = elbo_fn(self.x_train[random_index_train], self.y_train[random_index_train])
                test_elbo, test_kl = elbo_fn(self.x_test[random_index_test], self.y_test[random_index_test])

                # print output to screen
                print('Iteration: {} | Train ELBO: {:.4f} | Train KL {:.4f} | Test ELBO: {:.4f} | Test KL: {:.4f} | time: {:.4f} seconds'.format(i,
                                                                                                                                                 np.asscalar(train_elbo),
                                                                                                                                                 np.asscalar(train_kl),
                                                                                                                                                 np.asscalar(test_elbo),
                                                                                                                                                 np.asscalar(test_kl),
                                                                                                                                                 toc - tic))

                # start clock again
                tic = time.clock()

            # get statistics for test and train data
            if val_freq is not None and i % val_freq == 0:
                # validation
                # shuffle index for the validation data
                shuffled_index_ = np.random.choice(self.test_size, size=(200))
                # get new batches
                x_batch_val = self.x_test[shuffled_index_]
                y_batch_val = self.y_test[shuffled_index_]

                # get validation elbo and KL
                val_elbo, val_kl = elbo_fn(x_batch_val, y_batch_val)

                # train
                # shuffle index for the validation data
                shuffled_index_ = np.random.choice(self.train_size, size=(200))
                # get new batches
                x_batch_train = self.x_train[shuffled_index_]
                y_batch_train = self.y_train[shuffled_index_]

                # get validation elbo and KL
                train_elbo, train_kl = elbo_fn(x_batch_train, y_batch_train)

                # append iteration count
                iter_list.append(i)

                # elbo and kl
                elbo_train_list.append(train_elbo)
                elbo_val_list.append(val_elbo)
                KL_train_list.append(train_kl)
                KL_val_list.append(val_kl)

                # generate output from latent space using prior
                output_prior = self.call_generate_output_prior(generate_output_prior_x, generate_output_prior_y)
                # print to screen
                print('============')
                print('Output Prior')
                print('============')
                self.print_output_prior(output_prior, language_x='EN', language_y='FR')

                # randomized batches for posterior
                # shuffle index for the validation data

                # generate output from latent space using posterior, both for train and test set to see difference
                output_posterior_train = self.call_generate_output_posterior(generate_output_posterior_x,
                                                                             generate_output_posterior_y,
                                                                             x_batch_train[0: 4],
                                                                             y_batch_train[0: 4])
                output_posterior_val = self.call_generate_output_posterior(generate_output_posterior_x,
                                                                           generate_output_posterior_y,
                                                                           x_batch_val[0: 4],
                                                                           y_batch_val[0: 4])

                # print the output generated
                print('================')
                print('Output Posterior')
                print('================')
                print('Train')
                self.print_output_posterior(output_posterior_train, language_x='EN', language_y='FR')
                print('Validation')
                self.print_output_posterior(output_posterior_val, language_x='EN', language_y='FR')
                # append all to list
                # output_prior_list.append(output_prior)
                # output_posterior_test_list.append(output_posterior_val)
                # output_posterior_train_list.append(output_posterior_train)

            # save parameters (should check if we want to do early stopping, depends if we are in danger
            # of overfitting, should also check if we use dropout or other regularisation techniques tog
            # prevent overfitting.
            if save_params_every is not None and i % save_params_every == 0 and i > 0:
                with open(os.path.join(self.out_dir, 'gen_params_x.save'), 'wb') as f:
                    pickle.dump(self.sgvb.generative_model_x.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(self.out_dir, 'gen_params_y.save'), 'wb') as f:
                    pickle.dump(self.sgvb.generative_model_y.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
                    pickle.dump(self.sgvb.recognition_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(self.out_dir, 'all_embeddings_x.save'), 'wb') as f:
                    pickle.dump(self.sgvb.all_embeddings_x.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(self.out_dir, 'all_embeddings_y.save'), 'wb') as f:
                    pickle.dump(self.sgvb.all_embeddings_y.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
                    pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

                # save all lists of statistics as a dictionary
                save_dict = dict(elbo_train_list=elbo_train_list,
                                 elbo_val_list=elbo_val_list,
                                 KL_train_list=KL_train_list,
                                 KL_val_list=KL_val_list,
                                 output_prior_list=output_prior_list,
                                 output_posterior_train_list=output_posterior_train_list,
                                 output_posterior_test_list=output_posterior_test_list,
                                 iter_list=iter_list)

                with open(os.path.join(self.out_dir, 'statistics.save'), 'wb') as f:
                    pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            # increment the batch counter by one
            epoch_batch_counter += 1

        # save final parameters after training
        with open(os.path.join(self.out_dir, 'gen_params_x.save'), 'wb') as f:
            pickle.dump(self.sgvb.generative_model_x.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'gen_params_y.save'), 'wb') as f:
            pickle.dump(self.sgvb.generative_model_y.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
            pickle.dump(self.sgvb.recognition_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'all_embeddings_x.save'), 'wb') as f:
            pickle.dump(self.sgvb.all_embeddings_x.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'all_embeddings_y.save'), 'wb') as f:
            pickle.dump(self.sgvb.all_embeddings_y.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save all lists of statistics as a dictionary
        save_dict = dict(elbo_train_list=elbo_train_list,
                         elbo_val_list=elbo_val_list,
                         KL_train_list=KL_train_list,
                         KL_val_list=KL_val_list,
                         output_prior_list=output_prior_list,
                         output_posterior_train_list=output_posterior_train_list,
                         output_posterior_test_list=output_posterior_test_list,
                         iter_list=iter_list)

        with open(os.path.join(self.out_dir, 'statistics.save'), 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def test(self, batch_size, num_samples, sub_sample_size=None):
        """Get the test score for the test set

        :param batch_size: batch size
        :param num_samples: number of samples
        :sub_sample_size: number of subsampling done
        """
        elbo_fn = self.sgvb.elbo_fn(num_samples) if sub_sample_size is None else self.sgvb.elbo_fn(sub_sample_size)
        elbo = 0
        kl = 0

        batches_complete = 0

        # chunk X into batches (not sure why we would chunk it like that)
        for batch_X, batch_Y in zip(chunker([self.x_test], batch_size), chunker([self.y_test], batch_size)):
            batch_X = np.squeeze(batch_X)
            batch_Y = np.squeeze(batch_Y)
            start = time.clock()

            if sub_sample_size is None:
                elbo_batch, kl_batch = elbo_fn(batch_X, batch_Y)
            else:
                elbo_batch = 0
                kl_batch = 0

                for sub_sample in range(1, (num_samples / sub_sample_size) + 1):
                    elbo_sub_batch, kl_sub_batch = elbo_fn(batch_X, batch_Y)
                    elbo_batch = (elbo_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
                                                float(sub_sample * sub_sample_size))) + \
                                 (elbo_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))
                    kl_batch = (kl_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
                                            float(sub_sample * sub_sample_size))) + \
                               (kl_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))

            elbo += elbo_batch
            kl += kl_batch

            batches_complete += 1

            print('Tested batches ' + str(batches_complete) + ' of ' + str(round(self.x_test.shape[0] / batch_size))
                  + '; test set ELBO so far = ' + str(elbo) + ' (' + str(kl) + ')'
                  + ' / ' + str(elbo / (batches_complete * batch_size)) + ' ('
                  + str(kl / (batches_complete * batch_size)) + ') per obs.'
                  + ' (time taken = ' + str(time.clock() - start) + ' seconds)')

        print('Test set ELBO = ' + str(elbo))

    def generate_output(self, prior, posterior, num_outputs):
        """Generate output and save it to .npy files"""
        if prior:
            generate_output_prior_x, generate_output_prior_y = self.sgvb.generate_output_prior_fn(num_outputs, only_final=False)
            output_prior = self.call_generate_output_prior(generate_output_prior_x, generate_output_prior_y)

            for key, value in output_prior.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

        if posterior:
            generate_output_posterior_x, generate_output_posterior_y = self.sgvb.generate_output_posterior_fn(num_outputs, only_final=False)

            batch_indices = np.random.choice(len(self.x_train), num_outputs, replace=False)
            batch_in_X = np.array([self.x_train[ind] for ind in batch_indices]).astype(np.float32)
            batch_in_Y = np.array([self.y_train[ind] for ind in batch_indices]).astype(np.float32)

            output_posterior = self.call_generate_output_posterior(generate_output_posterior_x, generate_output_posterior_y, batch_in_X, batch_in_Y)

            for key, value in output_posterior.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

    def follow_latent_trajectory(self, num_samples, num_steps):
        [follow_latent_trajectory_x, follow_latent_trajectory_y] = self.sgvb.follow_latent_trajectory_fn(num_samples)
        step_size = 1. / (num_steps - 1)
        alphas = np.arange(0., 1. + step_size, step_size)
        chars_x, probs_x = follow_latent_trajectory_x(alphas)
        chars_y, probs_y = follow_latent_trajectory_y(alphas)

        out = OrderedDict()
        out['follow_traj_X_viterbi'] = chars_x
        out['follow_traj_X_probs_viterbi'] = probs_x
        out['follow_traj_Y_viterbi'] = chars_y
        out['follow_traj_Y_probs_viterbi'] = probs_y

        for key, value in out.items():
            np.save(os.path.join(self.out_dir, key + '.npy'), value)


class RunWords(RunTranslation):
    def __init__(self,
                 solver,
                 solver_kwargs,
                 recognition_model,
                 generative_model,
                 valid_vocab_x,
                 valid_vocab_y,
                 valid_vocab_counts_x,
                 valid_vocab_counts_y,
                 out_dir,
                 dataset_path_x,
                 dataset_path_y,
                 load_param_dir=None,
                 restrict_min_length=None,
                 restrict_max_length=None,
                 train_prop=0.95,
                 beam_size=5):
        """
        :param solver: solver class that handles sgvb training and updating
        :param solver_kwargs: kwargs for solver
        :param recognition_model: instance of the recognition model class
        :param generative_model: instance of the generative model class
        :param valid_vocab_x: valid vocabulary for x
        :param valid_vocab_y: valid vocabulary for y
        :param valid_vocab_counts_x: counts of each of the words in the valid vocabulary for x
        :param valid_vocab_counts_y: counts of each of the words in the valid vocabulary for y
        :param out_dir: path to out directory
        :param dataset_path_x: path to dataset of x
        :param dataset_path_y: path to dataset of y
        :param load_param_dir: path to directory of saved variables. If None, train from start
        :param restricted_min_length: restrict the min lengths of the sentences
        :param restricted_max_length: restrict the max lengths of the sentences
        :param train_prop: how much of the original data should be split into training/test set
        """
        # set all attributes
        self.solver_kwargs = solver_kwargs
        self.recognition_model = recognition_model
        self.generative_model = generative_model
        self.valid_vocab_x = valid_vocab_x
        self.valid_vocab_y = valid_vocab_y
        self.valid_vocab_counts_x = np.array(valid_vocab_counts_x)
        self.valid_vocab_counts_y = np.array(valid_vocab_counts_y)
        self.out_dir = out_dir
        self.dataset_path_x = dataset_path_x
        self.dataset_path_y = dataset_path_y
        self.load_param_dir = load_param_dir
        self.train_prop = train_prop
        self.beam_size = beam_size

        # data sets
        self.x_train, self.x_test, self.y_train, self.y_test, self.L_x_train, self.L_x_test, self.L_y_train, self.L_y_test = self.load_data(train_prop, restrict_min_length, restrict_max_length)

        print('All data sets loaded')
        print('#data points (train): {}, #data points (Test): {}'.format(len(self.L_x_train), len(self.L_x_test)))

        # Number of training and test examples
        # Might need to use validation dataset as well
        self.train_size = len(self.L_x_train)
        self.test_size = len(self.L_x_test)

        # # max_length from the actual data set and instantiate the solver
        self.max_length_x = np.concatenate((self.x_train, self.x_test), axis=0).shape[1]
        self.max_length_y = np.concatenate((self.y_train, self.y_test), axis=0).shape[1]
        # self.sgvb = solver(max_length=self.max_length, **self.solver_kwargs)

        print('Maximum length of sentence (x, y): ({}, {})'.format(self.max_length_x, self.max_length_x))

        # initialise sgvb solver
        self.sgvb = solver(max_len_x=self.max_length_x,
                           max_len_y=self.max_length_y,
                           vocab_count_x=self.valid_vocab_counts_x,
                           vocab_count_y=self.valid_vocab_counts_y,
                           **self.solver_kwargs)

        # if pretrained, load saved parameters of the model and set
        # the parameters of the recognition/generative models
        if load_param_dir is not None:
            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.sgvb.recognition_model.set_param_values(pickle.load(f))
            with open(os.path.join(self.load_param_dir, 'gen_params_x.save'), 'rb') as f:
                self.sgvb.generative_model_x.set_param_values(pickle.load(f))
            with open(os.path.join(self.load_param_dir, 'gen_params_y.save'), 'rb') as f:
                self.sgvb.generative_model_y.set_param_values(pickle.load(f))
            with open(os.path.join(self.load_param_dir, 'all_embeddings_x.save'), 'rb') as f:
                self.sgvb.all_embeddings_x.set_value(pickle.load(f))
            with open(os.path.join(self.load_param_dir, 'all_embeddings_y.save'), 'rb') as f:
                self.sgvb.all_embeddings_y.set_value(pickle.load(f))
            print('Parameters loaded and set.')

    def load_data(self, train_prop, restrict_min_length, restrict_max_length):
        """Load data set to use for training and testing

        :param train_prop: (float) float in [0, 1] indicating proportion of train/test split
        :param restrict_min_length: (int) lower restriction on the min lengths of sentences
        :param restrict_max_length: (int) upper restriction on the max lengths of sentences"""
        # We load the lists from the pickle files
        # datasets is of the form of list of lists,
        # each list consist of numbers from index of the
        # vocabulary. So N * max(L) list of lists of int.
        with open(self.dataset_path_x) as f:
            dataset_x = pickle.load(f)
        with open(self.dataset_path_y) as f:
            dataset_y = pickle.load(f)

        # words are interpreted abstractly (can be chars or words)
        words_x = []
        words_y = []

        # iterate over sentences
        if restrict_min_length is not None and restrict_max_length is not None:
            for sent_x, sent_y in zip(dataset_x, dataset_y):
                # filter out the sentences that are longer than restrict_max_length
                len_x = len(sent_x)
                len_y = len(sent_y)
                if len_x >= restrict_min_length and len_y >= restrict_min_length and len_x <= restrict_max_length and len_y <= restrict_max_length:
                    words_x.append(sent_x)
                    words_y.append(sent_y)
        else:
            words_x = dataset_x
            words_y = dataset_y

        # lengths of all of the words in source and target dataset
        L_x = np.array([len(sent_x) for sent_x in words_x])
        L_y = np.array([len(sent_y) for sent_y in words_y])

        # Numpy broadcasting to create a mask N * max(L)
        # the mask is such that it is True when the index
        # has a valid character, False when the original sentence
        # is done (When we have gone into the padding)
        pad_x = L_x[:, None] > np.arange(max(L_x))
        pad_y = L_y[:, None] > np.arange(max(L_y))

        # padd the sentences with zeros after they have ended
        words_to_return_x = np.full(pad_x.shape, 0, dtype='int')
        words_to_return_x[pad_x] = np.concatenate(words_x)
        words_to_return_y = np.full(pad_y.shape, 0, dtype='int')
        words_to_return_y[pad_y] = np.concatenate(words_y)

        # split the train/test data
        split = int(len(words_x) * train_prop)
        print(split, len(words_x), int(len(words_x) * train_prop))

        # return objects. Train and test set for x and y and lengths
        x_train = words_to_return_x[0: split]
        x_test = words_to_return_x[split:]
        y_train = words_to_return_y[0: split]
        y_test = words_to_return_y[split:]
        L_x_train = L_x[0: split]
        L_x_test = L_x[split:]
        L_y_train = L_y[0: split]
        L_y_test = L_y[split:]

        return x_train, x_test, y_train, y_test, L_x_train, L_x_test, L_y_train, L_y_test

    def call_generate_output_prior(self, generate_output_prior_x, generate_output_prior_y):
        """Call the generate_output_prior function and collect the output

        Works with x or y depending on which function you pass to it.

        :param generate_output_prior: (function) generates the output from WaveNet

        :return out: (OrderedDict) dictionary of all the relevant quantities"""
        z, x_gen = generate_output_prior_x()
        z, y_gen = generate_output_prior_y()

        out = OrderedDict()
        out['generated_z_prior'] = z
        out['generated_x_prior'] = x_gen
        out['generated_y_prior'] = y_gen

        return out

    def print_output_prior(self, output_prior, language_x, language_y):
        """Print the output from the prior on z

        Print the output as generated form the call_generate_output_prior
        function. Agnostic with respect to x or y, but have to be consistent
        in terms of vocabulary.

        :param output_prior: (OrderedDict) dictionary of the quantities from prior output
        :param valid_vocab: (string) the valid vocabulary for the generated output
        :param language: (str) language string, to show what language is being output"""
        # The sentences generated by viterbi (most probably sentences)
        x_gen = output_prior['generated_x_prior']
        y_gen = output_prior['generated_y_prior']

        print('='*10)

        # print generated sequences from  viterbi
        for n in range(x_gen.shape[0]):
            print(u'{} gen x (beam = {}): {}'.format(language_x,
                                                     self.beam_size,
                                                     u' '.join([self.valid_vocab_x[int(i)] for i in x_gen[n]])))
            print(u'{} gen y (beam = {}): {}'.format(language_y,
                                                     self.beam_size,
                                                     u' '.join([self.valid_vocab_y[int(i)] for i in y_gen[n]])))

            print('-'*10)

        print('='*10)

    def call_generate_output_posterior(self, generate_output_posterior_x, generate_output_posterior_y, x, y):
        """Call the generate_output_posterior function and collect the output

        Works with x or y depending on which function you pass to it.

        :param generate_output_prior: (function) generates the output from AUTR
        :param x: (np.array) batch input from language x
        :param y: (np.array) batch input from language y

        :return out: (OrderedDict) dictionary of all the relevant quantities"""
        z, x_gen = generate_output_posterior_x(x, y)
        z, y_gen = generate_output_posterior_y(y, x)

        out = OrderedDict()
        out['generated_z_posterior'] = z
        out['true_x_posterior'] = x
        out['generated_x_posterior'] = x_gen
        out['true_y_posterior'] = y
        out['generated_y_posterior'] = y_gen

        return out

    def print_output_posterior(self, output_posterior, language_x, language_y):
        """Print the output from the posterior on z

        Print the output as generated form the call_generate_output_posterior
        function. Agnostic with respect to x or y, but have to be consistent
        in terms of vocabulary.

        :param output_posterior: (OrderedDict) dictionary of the quantities from prior output
        :param valid_vocab: (string) the valid vocabulary for the generated output
        :param language: (str) language string, to show what language is being output"""
        x = output_posterior['true_x_posterior']
        x_gen = output_posterior['generated_x_posterior']
        y = output_posterior['true_y_posterior']
        y_gen = output_posterior['generated_y_posterior']

        valid_vocab_for_true_x = self.valid_vocab_x + [u'']
        valid_vocab_for_true_y = self.valid_vocab_y + [u'']

        print('='*10)

        # print generated posterior sequences from viterbi
        for n in range(x.shape[0]):
            print(language_x)
            print(u'true x : {}'.format(u' '.join([valid_vocab_for_true_x[i] for i in x[n]]).strip()))
            print(u'gen x  : {}'.format(u' '.join([self.valid_vocab_x[int(i)] for i in x_gen[n]])))
            print(language_y)
            print(u'true y : {}'.format(u' '.join([valid_vocab_for_true_y[i] for i in y[n]]).strip()))
            print(u'gen y  : {}'.format(u' '.join([self.valid_vocab_y[int(i)] for i in y_gen[n]])))
            print('-'*10)

        print('='*10)

    def train(self,
              num_iter,
              batch_size,
              num_samples,
              approximate_by_css,
              css_num_samples,
              grad_norm_constraint,
              update=adam,
              update_kwargs=None,
              warm_up=None,
              val_freq=None,
              val_num_samples=5,
              val_gen_print_samples=5,
              save_params_every=None):
        """Train the model

        :param num_iter: number of iterations to run for
        :param batch_size: batch_size
        :param num_samples: number of samples for elbo
        :param approximate_by_css:
        :param css_num_samples:
        :param grad_norm_constraint: gradient constraints
        :param update: update function
        :param update_kwargs: kwargs for update function
        :param warm_up: if warm up
        :param val_freq: validation frequency
        :param val_num_samples: number of validation samples
        :param val_gen_print_samples: how many generated validation sentences to print
        :param save_params_every: save parameters every _'th iteration"""
        if self.load_param_dir is not None:
            # Load parameters
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = pickle.load(f)
            with open(os.path.join(self.load_param_dir, 'statistics.save'), 'rb') as f:
                statistics = pickle.load(f)

            # Warm up already done
            warm_up = None

            # Reset the new things
            iter_train_list = statistics['iter_train']
            iter_test_list = statistics['iter_test']
            elbo_train_list = statistics['elbo_train']
            elbo_test_list = statistics['elbo_test']
            kl_train_list = statistics['kl_train']
            kl_test_list = statistics['kl_test']

            # Increment iter so that it is correct in the lists
            iter_count = iter_train_list[-1]
        else:
            saved_update = None
            # lists for output
            iter_train_list = []
            iter_test_list = []
            elbo_train_list = []
            elbo_test_list = []
            kl_train_list = []
            kl_test_list = []
            iter_count = 0

        # quick function for drawing random datapoints
        def new_batch(x, y, size, batch_size):
            # shuffle index for the validation data
            index = np.random.choice(size, batch_size, replace=False)
            # get new batches
            x_batch = x[index]
            y_batch = y[index]

            return x_batch, y_batch

        # Create optimiser which is the optimiser
        # function and the updates that comes from it
        print("Pre-optimiser")
        optimiser, updates = self.sgvb.optimiser(num_samples=num_samples,
                                                 approximate_by_css=approximate_by_css,
                                                 css_num_samples=css_num_samples,
                                                 grad_norm_constraint=grad_norm_constraint,
                                                 update=update,
                                                 update_kwargs=update_kwargs,
                                                 saved_update=saved_update)

        print("Optimiser, updates")

        # the actual elbo function that we are trying to optimise
        elbo_fn = self.sgvb.elbo_fn(val_num_samples, approximate_by_css=approximate_by_css, css_num_samples=css_num_samples)

        print("Elbo function")

        # the symbolic functions which generate output from the latent
        generate_output_prior_x, generate_output_prior_y = self.sgvb.generate_output_prior_fn(val_gen_print_samples)
        generate_output_posterior_x, generate_output_posterior_y = self.sgvb.generate_output_posterior_fn(val_gen_print_samples)

        print("Gen priors and posteriors")

        # if we are done
        done = False

        # Iterate through epochs, although we still count iterations
        while not done:
            print("Starting epoch")
            # get starting time
            tic = time.clock()
            # iterate the number of iterations needed per epoch
            iter_epochs = self.train_size/batch_size
            # shuffle index for this epoch
            shuffled_index = np.random.permutation(self.train_size)
            # iterate over batch
            for i in range(iter_epochs):
                x_batch = self.x_train[shuffled_index[i * batch_size: (i + 1) * batch_size]]
                y_batch = self.y_train[shuffled_index[i * batch_size: (i + 1) * batch_size]]

                # annealing constant (beta is a linear up to warm_up, where it is constant 1)
                if warm_up is not None:
                    beta = min(1.0, float(iter_count) / warm_up)
                else:
                    beta = 1.0

                # run the training step and get the approximate loss function for
                # this step. KL tells us how much of x, y is encoded in z. The
                # bigger KL is the more we are using the latent space.
                elbo, kl, log_p_x, log_p_y = optimiser(x_batch, y_batch, beta)
                elbo /= batch_size
                kl /= batch_size
                log_p_x /= batch_size
                log_p_y /= batch_size

                iter_train_list.append(iter_count)
                elbo_train_list.append(elbo)
                kl_train_list.append(kl)

                if iter_count % 50 == 0:
                    x_batch_val, y_batch_val = new_batch(self.x_test, self.y_test, self.test_size, batch_size)
                    val_elbo, val_kl, val_log_p_x, val_log_p_y = elbo_fn(x_batch_val, y_batch_val)
                    val_elbo /= batch_size
                    val_kl /= batch_size
                    val_log_p_x /= batch_size
                    val_log_p_y /= batch_size

                    iter_test_list.append(iter_count)
                    elbo_test_list.append(val_elbo)
                    kl_test_list.append(val_kl)

                    print('Test ELBO = {:.4f} (KL = {:.4f}) per data point'.format(np.asscalar(val_elbo),
                                                                                   np.asscalar(val_kl)))
                    print('Test log_p_x = {:.4f}, log_p_y = {:.4f}'.format(np.asscalar(val_log_p_x), np.asscalar(val_log_p_y)))

                toc = time.clock()
                print('Iteration {}: ELBO = {:.4f} (KL = {:.4f}) per data point (time taken = {:.4f}) seconds)'.format(iter_count,
                                                                                                                       np.asscalar(elbo),
                                                                                                                       np.asscalar(kl),
                                                                                                                       toc - tic))
                print('Iteration {}: LOG_P_X = {:.4f}, LOG_P_Y = {:.4f}, BETA = {:.4f}'.format(iter_count, np.asscalar(log_p_x),
                                                                                               np.asscalar(log_p_y), beta))
                tic = time.clock()

                # Output prior and posterior draws from language model
                if val_freq is not None and iter_count % val_freq == 0:
                    # generate output from latent space using prior
                    output_prior = self.call_generate_output_prior(generate_output_prior_x, generate_output_prior_y)
                    # print to screen
                    # todo Remove hardcoded language values
                    self.print_output_prior(output_prior, language_x='EN', language_y='FR')

                    # randomized batches for posterior
                    # shuffle index for the validation data
                    x_batch_val, y_batch_val = new_batch(self.x_test, self.y_test, self.test_size, val_gen_print_samples)
                    x_batch_train, y_batch_train = new_batch(self.x_train, self.y_train, self.train_size, val_gen_print_samples)

                    # generate output from latent space using posterior, both for train and test set to see difference
                    output_posterior_train = self.call_generate_output_posterior(generate_output_posterior_x,
                                                                                 generate_output_posterior_y,
                                                                                 x_batch_train,
                                                                                 y_batch_train)
                    output_posterior_val = self.call_generate_output_posterior(generate_output_posterior_x,
                                                                               generate_output_posterior_y,
                                                                               x_batch_val,
                                                                               y_batch_val)

                    # print the output generated
                    print('Train')
                    self.print_output_posterior(output_posterior_train, language_x='EN', language_y='FR')
                    print('Validation')
                    self.print_output_posterior(output_posterior_val, language_x='EN', language_y='FR')

                # increment iterations count by one
                iter_count += 1
                # breakout logic
                if iter_count >= num_iter:
                    done = True
                    break

                # save parameters (should check if we want to do early stopping, depends if we are in danger
                # of overfitting, should also check if we use dropout or other regularisation techniques tog
                # prevent overfitting.
                if save_params_every is not None and iter_count % save_params_every == 0 and iter_count > 0:
                    with open(os.path.join(self.out_dir, 'gen_params_x.save'), 'wb') as f:
                        pickle.dump(self.sgvb.generative_model_x.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(self.out_dir, 'gen_params_y.save'), 'wb') as f:
                        pickle.dump(self.sgvb.generative_model_y.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
                        pickle.dump(self.sgvb.recognition_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(self.out_dir, 'all_embeddings_x.save'), 'wb') as f:
                        pickle.dump(self.sgvb.all_embeddings_x.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(self.out_dir, 'all_embeddings_y.save'), 'wb') as f:
                        pickle.dump(self.sgvb.all_embeddings_y.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
                        pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

                    stats_dict = dict(iter_train=iter_train_list,
                                      iter_test=iter_test_list,
                                      elbo_train=elbo_train_list,
                                      elbo_test=elbo_test_list,
                                      kl_train=kl_train_list,
                                      kl_test=kl_test_list,
                                      warm_up=warm_up)

                    with open(os.path.join(self.out_dir, 'statistics.save'), 'wb') as f:
                        pickle.dump(stats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save final parameters after training
        with open(os.path.join(self.out_dir, 'gen_params_x.save'), 'wb') as f:
            pickle.dump(self.sgvb.generative_model_x.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'gen_params_y.save'), 'wb') as f:
            pickle.dump(self.sgvb.generative_model_y.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
            pickle.dump(self.sgvb.recognition_model.get_param_values(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'all_embeddings_x.save'), 'wb') as f:
            pickle.dump(self.sgvb.all_embeddings_x.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'all_embeddings_y.save'), 'wb') as f:
            pickle.dump(self.sgvb.all_embeddings_y.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            pickle.dump(updates, f, protocol=pickle.HIGHEST_PROTOCOL)

        stats_dict = dict(iter_train=iter_train_list,
                          iter_test=iter_test_list,
                          elbo_train=elbo_train_list,
                          elbo_test=elbo_test_list,
                          kl_train=kl_train_list,
                          kl_test=kl_test_list,
                          warm_up=warm_up)

        with open(os.path.join(self.out_dir, 'statistics.save'), 'wb') as f:
            pickle.dump(stats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    def test(self, batch_size, approximate_by_css, css_num_samples, num_samples, sub_sample_size=None, translate=False, translation_source=None):
        """Get the test score for the test set

        :param batch_size: batch size
        :param num_samples: number of samples
        :sub_sample_size: number of subsampling done
        """
        print("Create graph")
        if sub_sample_size is None:
            elbo_fn = self.sgvb.elbo_fn(num_samples, approximate_by_css=approximate_by_css, css_num_samples=css_num_samples,
                                        translate=translate, translation_source=translation_source)
        else:
            elbo_fn = self.sgvb.elbo_fn(sub_sample_size, approximate_by_css=approximate_by_css, css_num_samples=css_num_samples,
                                        translate=translate, translation_source=translation_source)

        elbo = 0
        kl = 0
        log_p_x = 0
        log_p_y = 0

        # Dimensions
        N, D = self.x_test.shape

        # Split up in terms of batch size
        num_batches = int(N / batch_size)

        print(num_batches)

        # iterate over all of the batches
        for i in range(num_batches):
            start = time.clock()
            batch_x = self.x_test[i * batch_size: (i + 1) * batch_size]
            batch_y = self.y_test[i * batch_size: (i + 1) * batch_size]
            elbo_batch, kl_batch, log_p_x_batch, log_p_y_batch = elbo_fn(batch_x, batch_y)
            end = time.clock()

            elbo += elbo_batch
            kl += kl_batch
            log_p_x += log_p_x_batch
            log_p_y += log_p_y_batch

            print('Processing batch {} took {:.4f} seconds'.format(i + 1, end - start))
            print('Percentage of batches completed: {}%'.format(100 * float(i + 1)/float(num_batches)))
            print('Current ELBO: {:.4f}, KL: {:.4f}'.format(float(elbo)/((i + 1) * batch_size), float(kl)/((i + 1) * batch_size)))
            print('Current LOG_P_X: {:.4f}, LOG_P_Y: {:.4f}'.format(float(log_p_x)/((i + 1) * batch_size), float(log_p_y)/((i + 1) * batch_size)))

        print('Final ELBO: {:.4f}, KL: {:.4f}'.format(float(elbo)/N, float(kl)/N))
        print('Final LOG_P_X: {:.4f}, LOG_P_Y: {:.4f}'.format(float(log_p_x)/N, float(log_p_y)/N))

    def generate_output(self, prior, posterior, num_outputs):
        """Generate output and save it to .npy files"""
        generated_dir = os.path.join(self.out_dir, 'generated_output')

        # if output directory doesn't exist
        if not os.path.exists(generated_dir):
            os.makedirs(generated_dir)

        if prior:
            generate_output_prior_x, generate_output_prior_y = self.sgvb.generate_output_prior_fn(num_outputs, self.beam_size)
            output_prior = self.call_generate_output_prior(generate_output_prior_x, generate_output_prior_y)

            for key, value in output_prior.items():
                np.save(os.path.join(generated_dir, key + '.npy'), value)

        if posterior:
            # 1 is the number of samples
            generate_output_posterior_x, generate_output_posterior_y = self.sgvb.generate_output_posterior_fn(1, self.beam_size)

            batch_indices = np.random.choice(len(self.x_train), num_outputs, replace=False)
            batch_in_x = np.array([self.x_train[ind] for ind in batch_indices]).astype(np.float32)
            batch_in_y = np.array([self.y_train[ind] for ind in batch_indices]).astype(np.float32)

            output_posterior = self.call_generate_output_posterior(generate_output_posterior_x, generate_output_posterior_y, batch_in_x, batch_in_y)

            for key, value in output_posterior.items():
                np.save(os.path.join(generated_dir, key + '.npy'), value)

    def generate_prior(self, num_outputs, beam_size):
        """Generate output prior.

        The output will be generated into a folder called generation"""
        prior_dir = os.path.join(self.out_dir, 'generation')

        # if output directory doesn't exist
        if not os.path.exists(prior_dir):
            os.makedirs(prior_dir)

        num_samples = 5
        num_iter = int(num_outputs/num_samples)

        prior_output_list_x = []
        prior_output_list_y = []
        prior_z_sampled_list = []

        ## Reconstruction
        # Prior
        generate_output_prior_x, generate_output_prior_y = self.sgvb.generate_output_prior_beam_fn(num_samples, beam_size)

        # Fill out list with generated sentences
        for i in range(num_iter):
            print('Iteration {} of {} of generating priors'.format(i+1, num_iter))
            z, x_gen_beam = generate_output_prior_x()
            z, y_gen_beam = generate_output_prior_y()
            prior_output_list_x.extend(x_gen_beam.tolist())
            prior_output_list_y.extend(y_gen_beam.tolist())
            prior_z_sampled_list.extend(z.tolist())

        prior_output_array_x = np.array(prior_output_list_x)
        prior_output_array_y = np.array(prior_output_list_y)
        prior_z_sampled_array = np.array(prior_z_sampled_list)

        # Save these to npy files
        np.save(os.path.join(prior_dir, 'generated_x_prior_beam.npy'), prior_output_array_x)
        np.save(os.path.join(prior_dir, 'generated_y_prior_beam.npy'), prior_output_array_y)
        np.save(os.path.join(prior_dir, 'prior_z_sampled.npy'), prior_z_sampled_array)

        prior_output_text_x = self.translate_one_hot_to_words(prior_output_array_x, self.valid_vocab_x)
        prior_output_text_y = self.translate_one_hot_to_words(prior_output_array_y, self.valid_vocab_y)

        # Save the text equivalent in pickle files, these can be loaded by helper.py
        # and printed to screen to check how it looks like
        with open(os.path.join(prior_dir, 'generated_x_prior_beam_txt.pkl'), 'wb') as f:
            pickle.dump(prior_output_text_x, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(prior_dir, 'generated_y_prior_beam_txt.pkl'), 'wb') as f:
            pickle.dump(prior_output_text_y, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save .txt files that can be inspected
        with codecs.open(os.path.join(prior_dir, 'generated_x_prior_beam.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(prior_output_text_x):
                print('Iteration {} of {} for writing x to file'.format(i+1, len(prior_output_text_x)))
                f.write(u'{}\n'.format(line))
        with codecs.open(os.path.join(prior_dir, 'generated_y_prior_beam.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(prior_output_text_y):
                print('Iteration {} of {} for writing y to file'.format(i+1, len(prior_output_text_y)))
                f.write(u'{}\n'.format(line))

    def generate_posterior(self, num_outputs, beam_size):
        """Generate output posterior.

        The output will be generated into a folder called generation"""
        posterior_dir = os.path.join(self.out_dir, 'generation')

        # if output directory doesn't exist
        if not os.path.exists(posterior_dir):
            os.makedirs(posterior_dir)

        batch_size = 20
        num_iter = int(np.floor(num_outputs/batch_size))

        posterior_output_list_x_train = []
        posterior_output_list_y_train = []
        posterior_z_sampled_list_train = []
        true_x_train = []
        true_y_train = []
        posterior_output_list_x_test = []
        posterior_output_list_y_test = []
        posterior_z_sampled_list_test = []
        true_x_test = []
        true_y_test = []

        generate_output_posterior_x, generate_output_posterior_y = self.sgvb.generate_output_posterior_beam_fn(1, beam_size)

        # Fill out list with generated sentences
        for i in range(num_iter):
            print('Iteration {} of {} of generating posteriors'.format(i+1, num_iter))
            x_batch_train = self.x_train[i * batch_size: (i + 1) * batch_size]
            y_batch_train = self.y_train[i * batch_size: (i + 1) * batch_size]
            x_batch_test = self.x_test[i * batch_size: (i + 1) * batch_size]
            y_batch_test = self.y_test[i * batch_size: (i + 1) * batch_size]

            z_train, x_train_gen_beam = generate_output_posterior_x(x_batch_train, y_batch_train)
            z_train, y_train_gen_beam = generate_output_posterior_y(y_batch_train, x_batch_train)
            z_test, x_test_gen_beam = generate_output_posterior_x(x_batch_test, y_batch_test)
            z_test, y_test_gen_beam = generate_output_posterior_y(y_batch_test, x_batch_test)

            posterior_output_list_x_train.extend(x_train_gen_beam.tolist())
            posterior_output_list_y_train.extend(y_train_gen_beam.tolist())
            posterior_z_sampled_list_train.extend(z_train.tolist())
            true_x_train.extend(x_batch_train.tolist())
            true_y_train.extend(y_batch_train.tolist())

            posterior_output_list_x_test.extend(x_test_gen_beam.tolist())
            posterior_output_list_y_test.extend(y_test_gen_beam.tolist())
            posterior_z_sampled_list_test.extend(z_test.tolist())
            true_x_test.extend(x_batch_test)
            true_y_test.extend(y_batch_test)

        posterior_output_array_x_train = np.array(posterior_output_list_x_train)
        posterior_output_array_y_train = np.array(posterior_output_list_y_train)
        posterior_z_sampled_array_train = np.array(posterior_z_sampled_list_train)
        true_x_array_train = np.array(true_x_train)
        true_y_array_train = np.array(true_y_train)

        posterior_output_array_x_test = np.array(posterior_output_list_x_test)
        posterior_output_array_y_test = np.array(posterior_output_list_y_test)
        posterior_z_sampled_array_test = np.array(posterior_z_sampled_list_test)
        true_x_array_test = np.array(true_x_test)
        true_y_array_test = np.array(true_y_test)

        # Save these to npy files
        np.save(os.path.join(posterior_dir, 'generated_x_posterior_beam_train.npy'), posterior_output_array_x_train)
        np.save(os.path.join(posterior_dir, 'generated_y_posterior_beam_train.npy'), posterior_output_array_y_train)
        np.save(os.path.join(posterior_dir, 'posterior_z_sampled_train.npy'), posterior_z_sampled_array_train)
        np.save(os.path.join(posterior_dir, 'true_x_array_train.npy'), true_x_array_train)
        np.save(os.path.join(posterior_dir, 'true_y_array_train.npy'), true_y_array_train)

        np.save(os.path.join(posterior_dir, 'generated_x_posterior_beam_test.npy'), posterior_output_array_x_test)
        np.save(os.path.join(posterior_dir, 'generated_y_posterior_beam_test.npy'), posterior_output_array_y_test)
        np.save(os.path.join(posterior_dir, 'posterior_z_sampled_test.npy'), posterior_z_sampled_array_test)
        np.save(os.path.join(posterior_dir, 'true_x_array_test.npy'), true_x_array_test)
        np.save(os.path.join(posterior_dir, 'true_y_array_test.npy'), true_y_array_test)

        posterior_output_text_x_train = self.translate_one_hot_to_words(posterior_output_array_x_train, self.valid_vocab_x)
        posterior_output_text_y_train = self.translate_one_hot_to_words(posterior_output_array_y_train, self.valid_vocab_y)
        true_x_text_train = self.translate_one_hot_to_words(true_x_array_train, self.valid_vocab_x)
        true_y_text_train = self.translate_one_hot_to_words(true_y_array_train, self.valid_vocab_y)

        posterior_output_text_x_test = self.translate_one_hot_to_words(posterior_output_array_x_test, self.valid_vocab_x)
        posterior_output_text_y_test = self.translate_one_hot_to_words(posterior_output_array_y_test, self.valid_vocab_y)
        true_x_text_test = self.translate_one_hot_to_words(true_x_array_test, self.valid_vocab_x)
        true_y_text_test = self.translate_one_hot_to_words(true_y_array_test, self.valid_vocab_y)

        # Save the text equivalent in pickle files, these can be loaded by helper.py
        # and printed to screen to check how it looks like
        with open(os.path.join(posterior_dir, 'generated_x_posterior_beam_train_txt.pkl'), 'wb') as f:
            pickle.dump(posterior_output_text_x_train, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(posterior_dir, 'generated_y_posterior_beam_train_txt.pkl'), 'wb') as f:
            pickle.dump(posterior_output_text_y_train, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(posterior_dir, 'true_x_train_txt.pkl'), 'wb') as f:
            pickle.dump(true_x_text_train, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(posterior_dir, 'true_y_train_txt.pkl'), 'wb') as f:
            pickle.dump(true_y_text_train, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(posterior_dir, 'generated_x_posterior_beam_test_txt.pkl'), 'wb') as f:
            pickle.dump(posterior_output_text_x_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(posterior_dir, 'generated_y_posterior_beam_test_txt.pkl'), 'wb') as f:
            pickle.dump(posterior_output_text_y_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(posterior_dir, 'true_x_test_txt.pkl'), 'wb') as f:
            pickle.dump(true_x_text_test, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(posterior_dir, 'true_y_test_txt.pkl'), 'wb') as f:
            pickle.dump(true_y_text_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save .txt files that can be inspected
        with codecs.open(os.path.join(posterior_dir, 'generated_x_posterior_beam_train.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(posterior_output_text_x_train):
                print('Iteration {} of {} for writing x to file'.format(i+1, len(posterior_output_text_x_train)))
                f.write(u'{}\n'.format(line))
        with codecs.open(os.path.join(posterior_dir, 'generated_y_posterior_beam_train.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(posterior_output_text_y_train):
                print('Iteration {} of {} for writing y to file'.format(i+1, len(posterior_output_text_y_train)))
                f.write(u'{}\n'.format(line))
        with codecs.open(os.path.join(posterior_dir, 'true_x_train.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(true_x_text_train):
                print('Iteration {} of {} for writing x to file'.format(i+1, len(true_x_text_train)))
                f.write(u'{}\n'.format(line))
        with codecs.open(os.path.join(posterior_dir, 'true_y_train.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(true_y_text_train):
                print('Iteration {} of {} for writing y to file'.format(i+1, len(true_y_text_train)))
                f.write(u'{}\n'.format(line))

        with codecs.open(os.path.join(posterior_dir, 'generated_x_posterior_beam_test.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(posterior_output_text_x_test):
                print('Iteration {} of {} for writing x to file'.format(i+1, len(posterior_output_text_x_test)))
                f.write(u'{}\n'.format(line))
        with codecs.open(os.path.join(posterior_dir, 'generated_y_posterior_beam_test.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(posterior_output_text_y_test):
                print('Iteration {} of {} for writing y to file'.format(i+1, len(posterior_output_text_y_test)))
                f.write(u'{}\n'.format(line))
        with codecs.open(os.path.join(posterior_dir, 'true_x_test.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(true_x_text_test):
                print('Iteration {} of {} for writing x to file'.format(i+1, len(true_x_text_test)))
                f.write(u'{}\n'.format(line))
        with codecs.open(os.path.join(posterior_dir, 'true_y_test.txt'), 'w', 'utf8') as f:
            for i, line in enumerate(true_y_text_test):
                print('Iteration {} of {} for writing y to file'.format(i+1, len(true_y_text_test)))
                f.write(u'{}\n'.format(line))


    def follow_latent_trajectory(self, num_samples, num_steps):
        [follow_latent_trajectory_x, follow_latent_trajectory_y] = self.sgvb.follow_latent_trajectory_fn(num_samples)
        step_size = 1. / (num_steps - 1)
        alphas = np.arange(0., 1. + step_size, step_size)
        chars_x, probs_x = follow_latent_trajectory_x(alphas)
        chars_y, probs_y = follow_latent_trajectory_y(alphas)

        out = OrderedDict()
        out['follow_traj_X_viterbi'] = chars_x
        out['follow_traj_X_probs_viterbi'] = probs_x
        out['follow_traj_Y_viterbi'] = chars_y
        out['follow_traj_Y_probs_viterbi'] = probs_y

        for key, value in out.items():
            np.save(os.path.join(self.out_dir, key + '.npy'), value)

    def perplexity(self, num_datapoints=None, batch_size=100, num_samples=1):
        """Calculate the perplexity of the test dataset.

        Since we can't actually evaluate the log-likelihood
        directly which is needed to calculate the perplexity,
        we will approximate the log-likelihood

        p(x) \approx 1/S\sum_s^S p(x | z^s)

        and then do the necessary transformations.

        :param num_datapoints: (int) number of datapoints to evaluate over
        :param batch_size: (int) batch size
        :param num_samples: (int) number of samples of z
        """

        if num_datapoints is None or num_datapoints == 0:
            num_datapoints = self.test_x.shape[0]

        num_iter = np.floor(num_datapoints/batch_size) + 1

        # Log-likelihood approximate theano function
        elbo_x_fn = self.sgvb.elbo_fn(num_samples, approximate_by_css=False, css_num_samples=0, translate=True, translation_source='x')
        elbo_y_fn = self.sgvb.elbo_fn(num_samples, approximate_by_css=False, css_num_samples=0, translate=True, translation_source='y')
        elbo_fn = self.sgvb.elbo_fn(num_samples, approximate_by_css=False, css_num_samples=0, translate=False, translation_source='y')

        set_x = self.test_x[0: num_datapoints]
        set_y = self.test_y[0: num_datapoints]

        def new_batch(iteration):
            x_batch = set_x[iteration * batch_size, (iteration + 1) * batch_size]
            y_batch = set_y[iteration * batch_size, (iteration + 1) * batch_size]
            return x_batch, y_batch

        denom_pp_x = np.sum(self.L_x_test[0: num_datapoints])
        denom_pp_y = np.sum(self.L_y_test[0: num_datapoints])
        denom_pp = denom_pp_x + denom_pp_y

        elbo_x = 0
        elbo_y = 0
        elbo_sum = 0

        # Loop over perplexity calculation
        for i in range(num_iter):
            x_batch, y_batch = new_batch(i)
            _, kl_x, log_p_x, _ = elbo_x_fn(x_batch, y_batch)
            _, kl_y, log_p_y, _ = elbo_y_fn(x_batch, y_batch)
            elbo, kl, _, _ = elbo_fn(x_batch, y_batch)
            elbo_sum += elbo
            elbo_x_sum += log_p_x - kl_x
            elbo_y_sum += log_p_y - kl_y

        pp_x = np.exp(-(1.0/denom_pp_x) * elbo_x)
        pp_y = np.exp(-(1.0/denom_pp_y) * elbo_y)
        pp = np.exp(-(1.0/denom_pp) * elbo_sum)

        # Could check to do this using translation instead, to get the true ELBO_x and ELBO_y

        print('(ELBO) perplexity\nFull (x, y): {}, EN: {}, FR: {}'.format(pp, pp_x, pp_y))

    def translate(self, num_output=200, translation_source='x', word_dict_x=None, word_dict_y=None):
        """Use the form q(z | x) = q(z | x, y) in order to do translation

        We will save the output directly in the model folder, the output
        will then be preprocessed by the helper script.

        :param num_output: number of output from test set we will translate
        :param translation_source: original language to translate from, if 'x': x -> y, else y -> x"""
        # DEBUGGING STARTS HERE
        # generate_output_posterior_x, generate_output_posterior_y = self.sgvb.generate_output_posterior_fn(num_output)

        # batch_indices = np.random.choice(len(self.x_train), 20, replace=False)
        # print(batch_indices)
        # batch_in_x = np.array([self.x_train[ind] for ind in batch_indices]).astype(np.float32)
        # batch_in_y = np.array([self.y_train[ind] for ind in batch_indices]).astype(np.float32)

        # output_posterior = self.call_generate_output_posterior(generate_output_posterior_x, generate_output_posterior_y, batch_in_x, batch_in_y)
        # print(output_posterior)
        # # DEBUGGING ENDS HERE

        generate_output_posterior_x, generate_output_posterior_y = self.sgvb.generate_output_translation_fn(20, translation_source)

        true_x = []
        true_y = []
        generated_x = []
        generated_y = []
        z_sampled = []

        batch_size = 20

        # Number of iterations we need to produce the sentences
        if num_output is None:
            num_iter = int(np.floor(self.test_size/batch_size))
        else:
            num_iter = int(np.floor(num_output/batch_size))

        print('Number of batches needed: {}'.format(num_iter))

        # Loop over and output generated sentences
        for i in range(num_iter):
            batch_in_x = self.x_test[i * batch_size: (i + 1) * batch_size].astype(np.float32)
            batch_in_y = self.y_test[i * batch_size: (i + 1) * batch_size].astype(np.float32)

            output = self.call_generate_output_posterior(generate_output_posterior_x, generate_output_posterior_y, batch_in_x, batch_in_y)

            # Fill out lists
            generated_x.extend(output['generated_x_posterior'].tolist())
            generated_y.extend(output['generated_y_posterior'].tolist())
            true_x.extend(output['true_x_posterior'].tolist())
            true_y.extend(output['true_y_posterior'].tolist())
            z_sampled.extend(output['generated_z_posterior'].tolist())
            print('{} % Processed'.format(100 * float(i + 1)/num_iter))

        if num_output % batch_size != 0:
            # Final output
            batch_in_x = self.x_test[num_iter * batch_size: num_output].astype(np.float32)
            batch_in_y = self.y_test[num_iter * batch_size: num_output].astype(np.float32)
            print(batch_in_x.shape, batch_in_y.shape)
            output = self.call_generate_output_posterior(generate_output_posterior_x, generate_output_posterior_y, batch_in_x, batch_in_y)

            # Fill out lists
            generated_x.extend(output['generated_x_posterior'].tolist())
            generated_y.extend(output['generated_y_posterior'].tolist())
            true_x.extend(output['true_x_posterior'].tolist())
            true_y.extend(output['true_y_posterior'].tolist())
            z_sampled.extend(output['generated_z_posterior'].tolist())
        else:
            pass

        print('Everything processed')

        # Produce numpy arrays from lists
        generated_x = np.asarray(generated_x, dtype=np.int32)
        generated_y = np.asarray(generated_y, dtype=np.int32)
        true_x = np.asarray(true_x, dtype=np.int32)
        true_y = np.asarray(true_y, dtype=np.int32)
        z_sampled = np.asarray(z_sampled, dtype=np.int32)
        output = dict(generated_x=generated_x,
                      generated_y=generated_y,
                      true_x=true_x,
                      true_y=true_y,
                      z_sampled=z_sampled)

        for key, value in output.items():
            save_path = os.path.join(self.out_dir, key + '_translation.npy')
            np.save(save_path, value)
            print('Saved to file {}'.format(save_path))

                # Decode sentences
        if word_dict_x is None:
            word_dict_x = self.valid_vocab_x

        if word_dict_y is None:
            word_dict_y = self.valid_vocab_y

        true_x_sentences = self.translate_one_hot_to_words(true_x, word_dict_x)
        generated_x_sentences = self.translate_one_hot_to_words(generated_x, word_dict_x)

        true_y_sentences = self.translate_one_hot_to_words(true_y, word_dict_y)
        generated_y_sentences = self.translate_one_hot_to_words(generated_y, word_dict_y)

        # Create list of reference sentences
        true_x_list = []
        true_y_list = []
        generated_x_list = []
        generated_y_list = []

        for i in range(len(true_x_sentences)):
            true_x_list.append([word_tokenize(true_x_sentences[i].split('<EOS>', 1)[0].rstrip(' '))])
            generated_x_list.append(word_tokenize(generated_x_sentences[i].split('<EOS>', 1)[0].rstrip(' ')))

        for i in range(len(true_y_sentences)):
            true_y_list.append([word_tokenize(true_y_sentences[i].split('<EOS>', 1)[0].rstrip(' '))])
            generated_y_list.append(word_tokenize(generated_y_sentences[i].split('<EOS>', 1)[0].rstrip(' ')))

        # Compute BLEU score
        blue_x = corpus_bleu(true_x_list, generated_x_list)
        blue_y = corpus_bleu(true_y_list, generated_y_list)

        print('BLEU score: EN {:.4f} FR {:.4f}'.format(blue_x, blue_y))

    def translate_one_hot_to_words(self, array, word_dict):
        """Translate an array of one hot encoded words to strings"""
        N, L = array.shape
        array = array.astype(np.int32)
        sentences = []

        for i in range(N):
            sentence = []
            for j in range(L):
                sentence.append(word_dict[array[i, j]])
            sentence = ' '.join(sentence)
            sentences.append(sentence)

        return sentences
