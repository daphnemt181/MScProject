"""
Helper script for dealing with the created data
"""

from __future__ import print_function

import sys
import os
import time
from pprint import pprint
import cPickle as pickle
import codecs
import glob

sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../data/')

sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
#import seaborn as sns

from SETTINGS import *

class EvaluateRunReconstruction():
    """Evaluate all of the dumped data from run"""

    def __init__(self, save_path, dict_path):
        """
        :param save_path: (str) name of the final save directory
        :param dict_path: (str) name of the dictionary directory
        """
        self.dict_path = os.path.join(PATHS['data_processed_dir'], dict_path)
        self.save_path = os.path.join(PATHS['save_dir'], save_path)
        self.output_path = os.path.join(self.save_path, 'generated_analytics')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Get all of the data

        # Pickled data
        with open(os.path.join(self.save_path, 'allvars.save')) as f:
            # All variables at start of training (Including all hyperparameter settings)
            self.allvars = pickle.load(f)
        with open(os.path.join(self.dict_path, 'en_valid_vocab.pickle')) as f:
            # All variables at start of training (Including all hyperparameter settings)
            self.word_dict = pickle.load(f)
        with open(os.path.join(self.save_path, 'statistics.save')) as f:
            # All numpy arrays for iteration step, train/test ELBO/-KL
            # keys:
            # 'elbo_test'
            # 'iter_test'
            # 'warm_up'
            # 'elbo_train'
            # 'iter_train'
            # 'kl_test'
            # 'kl_train'
            self.statistics = pickle.load(f)

        # Numpy arrays
        self.generated_x_argmax_posterior = np.load(os.path.join(self.save_path, 'generated_x_argmax_posterior.npy'))
        self.generated_x_argmax_prior = np.load(os.path.join(self.save_path, 'generated_x_argmax_prior.npy'))
        self.true_x_for_posterior = np.load(os.path.join(self.save_path, 'true_x_for_posterior.npy'))
        self.generated_x_sampled_posterior = np.load(os.path.join(self.save_path, 'generated_x_sampled_posterior.npy'))
        self.generated_x_sampled_prior = np.load(os.path.join(self.save_path, 'generated_x_sampled_prior.npy'))
        self.generated_z_posterior = np.load(os.path.join(self.save_path, 'generated_z_posterior.npy'))
        self.generated_z_prior = np.load(os.path.join(self.save_path, 'generated_z_prior.npy'))

        # create pandas dataframes
        # Series
        elbo_train_ser = pd.Series(data=self.statistics['elbo_train'],
                                  index=self.statistics['iter_train'])
        elbo_test_ser = pd.Series(data=self.statistics['elbo_test'],
                                  index=self.statistics['iter_test'])
        kl_train_ser = pd.Series(data=self.statistics['kl_train'],
                                 index=self.statistics['iter_train'])
        kl_test_ser = pd.Series(data=self.statistics['kl_test'],
                                index=self.statistics['iter_test'])

        # train
        self.train_df =  pd.DataFrame(dict(elbo_train=elbo_train_ser,
                                           kl_train=kl_train_ser))
        self.train_df['elbo_train_ma'] = self.train_df['elbo_train'].rolling(window=100).mean()
        self.train_df['kl_train_ma'] = self.train_df['kl_train'].rolling(window=100).mean()

        # test
        self.test_df =  pd.DataFrame(dict(elbo_test=elbo_test_ser,
                                           kl_test=kl_test_ser))
        self.test_df['elbo_test_ma'] = self.test_df['elbo_test'].rolling(window=100).mean()
        self.test_df['kl_test_ma'] = self.test_df['kl_test'].rolling(window=100).mean()

        # Make into dataframe
        self.stats_df = pd.DataFrame(dict(elbo_train=elbo_train_ser,
                                          elbo_test=elbo_test_ser,
                                          kl_train=kl_train_ser,
                                          kl_test=kl_test_ser)).interpolate()

    def fast_plot(self):
        """Interactively create the plot inline"""
        self.stats_df.plot()

    def create_plots(self):
        """Create all of the necessary plots

        The plots will be saved to .eps files and several
        plots will be saved, depending on how you want to
        use them in the thesis."""
        # Set all styles
        plt.style.use('seaborn-dark')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['lines.linewidth'] = 1.2
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 12

        color_dict = dict(elbo_train='#ef4747',
                          elbo_test='#ef7347',
                          kl_train='#47efc7',
                          kl_test='#47caef')

        # Create first axis
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.plot(self.train_df['elbo_train'], color=color_dict['elbo_train'], alpha=1.0, label='ELBO (Train)')
        ax1.plot(self.test_df['elbo_test'], color=color_dict['elbo_test'], alpha=1.0, label='ELBO (Test)')
        ax1.set_ylim(-100, -45)
        ax1.set_xlabel('time steps')
        ax1.set_ylabel('ELBO', color=color_dict['elbo_train'])
        ax1.tick_params('y', colors=color_dict['elbo_train'])

        # Twin the second axis so we can plot KL and ELBO on same figure
        ax2 = ax1.twinx()
        ax2.plot(self.train_df['kl_train'], color=color_dict['kl_train'], alpha=1.0, label='KL (Train)')
        ax2.plot(self.test_df['kl_test'], color=color_dict['kl_test'], alpha=1.0, label='KL (Test)')
        ax2.set_ylim(0, 50)
        ax2.set_ylabel('KL', color=color_dict['kl_train'])
        ax2.tick_params('y', colors=color_dict['kl_train'])

        # Gather lines and labels to create legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True, fancybox=True, shadow=False)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')

        # Add warmup line
        ax2.axvline(self.statistics['warm_up'], color='k', linewidth=0.5)
        xy = (float(self.statistics['warm_up'])/self.statistics['iter_train'][-1] + 0.1, 0.3)
        plt.annotate(r'$\tau = {}$'.format(self.statistics['warm_up']),
                     xy=xy,
                     xycoords="axes fraction",
                     horizontalalignment='center',
                     verticalalignment='center')

        # Align y-axis ticks to that grid doesn't get superimposed for KL/ELBO
        l1 = ax1.get_ylim()
        l2 = ax2.get_ylim()
        f = lambda x : l2[0]+(x-l1[0])/(l1[1]-l1[0])*(l2[1]-l2[0])
        ticks = f(ax1.get_yticks())
        ax2.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))

        # Set grid, only for first axis to not write over second axis
        ax1.grid(False)

        # Tight layout
        fig.tight_layout()

        # Save it all
        save_img_path = os.path.join(self.output_path, 'full_fig')
        fig.savefig(save_img_path + '.eps', dpi=400, format='eps')
        fig.savefig(save_img_path + '.png', dpi=400, format='png')
        print('Image saved to path {}.(eps/png)'.format(save_img_path))

        plt.close("all")

    def output_sentences(self):
        """Output all of the generated sentences"""
        self.generated_x_argmax_posterior = np.load(os.path.join(self.save_path, 'generated_x_argmax_posterior.npy'))
        self.generated_x_argmax_prior = np.load(os.path.join(self.save_path, 'generated_x_argmax_prior.npy'))
        self.true_x_for_posterior = np.load(os.path.join(self.save_path, 'true_x_for_posterior.npy'))
        self.generated_x_sampled_posterior = np.load(os.path.join(self.save_path, 'generated_x_sampled_posterior.npy'))
        self.generated_x_sampled_prior = np.load(os.path.join(self.save_path, 'generated_x_sampled_prior.npy'))
        self.generated_z_posterior = np.load(os.path.join(self.save_path, 'generated_z_posterior.npy'))
        self.generated_z_prior = np.load(os.path.join(self.save_path, 'generated_z_prior.npy'))

        generated_x_argmax_posterior_sentences = translate_one_hot_to_words(self.generated_x_argmax_posterior, self.word_dict)
        generated_x_argmax_prior_sentences = translate_one_hot_to_words(self.generated_x_argmax_prior, self.word_dict)
        true_x_for_posterior_sentences = translate_one_hot_to_words(self.true_x_for_posterior, self.word_dict)
        generated_x_sampled_posterior_sentences = translate_one_hot_to_words(self.generated_x_sampled_posterior, self.word_dict)
        generated_x_sampled_prior_sentences = translate_one_hot_to_words(self.generated_x_sampled_prior, self.word_dict)
        # generated_z_posterior_sentences = translate_one_hot_to_words(self.generated_z_posterior, self.word_dict)
        # generated_z_prior_sentences = translate_one_hot_to_words(self.generated_z_prior, self.word_dict)

        sentence_dict = dict(generated_x_argmax_posterior_sentences=generated_x_argmax_posterior_sentences,
                             generated_x_argmax_prior_sentences=generated_x_argmax_prior_sentences,
                             true_x_for_posterior_sentences=true_x_for_posterior_sentences,
                             generated_x_sampled_posterior_sentences=generated_x_sampled_posterior_sentences,
                             generated_x_sampled_prior_sentences=generated_x_sampled_prior_sentences)
                             # generated_z_posterior_sentences=generated_z_posterior_sentences,
                             # generated_z_prior_sentences=generated_z_prior_sentences)

        for key in sentence_dict.keys():
            with codecs.open(os.path.join(self.output_path, key + '.txt'), 'w', 'utf8') as f:
                f.write("\n".join(sentence_dict[key]))
                print('Wrote {}.txt to file'.format(key))

        print('All generated output written to disk')

    def get_hyperparams(self):
        """Output all of the possible hyperparameter information+more"""
        pprint(self.allvars)


class EvaluateRunTranslation(EvaluateRunReconstruction):
    """Evaluate all of the dumped data from run"""

    def __init__(self, save_path, dict_path):
        self.dict_path = os.path.join(PATHS['data_processed_dir'], dict_path)
        self.save_path = os.path.join(PATHS['save_dir'], save_path)
        self.output_path = os.path.join(self.save_path, 'generated_analytics')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Get all of the data

        # Pickled data
        with open(os.path.join(self.save_path, 'allvars.save')) as f:
            # All variables at start of training (Including all hyperparameter settings)
            self.allvars = pickle.load(f)
        with open(os.path.join(self.dict_path, 'en_valid_vocab.pickle')) as f:
            # All variables at start of training (Including all hyperparameter settings)
            self.word_dict_en = pickle.load(f)
        with open(os.path.join(self.dict_path, 'fr_valid_vocab.pickle')) as f:
            # All variables at start of training (Including all hyperparameter settings)
            self.word_dict_fr = pickle.load(f)
        with open(os.path.join(self.save_path, 'statistics.save')) as f:
            # All numpy arrays for iteration step, train/test ELBO/KL
            # keys:
            # 'elbo_test'
            # 'iter_test'
            # 'warm_up'
            # 'elbo_train'
            # 'iter_train'
            # 'kl_test'
            # 'kl_train'
            self.statistics = pickle.load(f)

        # Numpy arrays
        # TODO: Get this working

        # create pandas dataframes
        # Series
        elbo_train_ser = pd.Series(data=self.statistics['elbo_train'],
                                   index=self.statistics['iter_train'], dtype=np.float32)
        elbo_test_ser = pd.Series(data=self.statistics['elbo_test'],
                                  index=self.statistics['iter_test'], dtype=np.float32)
        kl_train_ser = pd.Series(data=self.statistics['kl_train'],
                                 index=self.statistics['iter_train'], dtype=np.float32)
        kl_test_ser = pd.Series(data=self.statistics['kl_test'],
                                index=self.statistics['iter_test'], dtype=np.float32)

        # train
        self.train_df =  pd.DataFrame(dict(elbo_train=elbo_train_ser,
                                           kl_train=kl_train_ser))
        self.train_df['elbo_train_ma'] = self.train_df['elbo_train'].rolling(window=100).mean()
        self.train_df['kl_train_ma'] = self.train_df['kl_train'].rolling(window=100).mean()

        # test
        self.test_df =  pd.DataFrame(dict(elbo_test=elbo_test_ser,
                                           kl_test=kl_test_ser))
        self.test_df['elbo_test_ma'] = self.test_df['elbo_test'].rolling(window=100).mean()
        self.test_df['kl_test_ma'] = self.test_df['kl_test'].rolling(window=100).mean()


        # Make into dataframe
        self.stats_df = pd.DataFrame(dict(elbo_train=elbo_train_ser,
                                          elbo_test=elbo_test_ser,
                                          kl_train=kl_train_ser,
                                          kl_test=kl_test_ser))

    def create_plots(self):
        """Create all of the necessary plots

        The plots will be saved to .eps files and several
        plots will be saved, depending on how you want to
        use them in the thesis."""
        # Set all styles
        plt.style.use('seaborn-dark')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['lines.linewidth'] = 1.2
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 12

        color_dict = dict(elbo_train='#ef4747',
                          elbo_test='#ef7347',
                          kl_train='#47efc7',
                          kl_test='#47caef')

        # Create first axis
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.plot(self.train_df['elbo_train'], color=color_dict['elbo_train'], alpha=1.0, label='ELBO (Train)')
        ax1.plot(self.test_df['elbo_test'], color=color_dict['elbo_test'], alpha=1.0, label='ELBO (Test)')
        ax1.set_ylim(-300, max(self.train_df['elbo_train']) + 20)
        ax1.set_xlabel('time steps')
        ax1.set_ylabel('ELBO', color=color_dict['elbo_train'])
        ax1.tick_params('y', colors=color_dict['elbo_train'])

        # Twin the second axis so we can plot KL and ELBO on same figure
        ax2 = ax1.twinx()
        ax2.plot(self.train_df['kl_train'], color=color_dict['kl_train'], alpha=1.0, label='KL (Train)')
        ax2.plot(self.test_df['kl_test'], color=color_dict['kl_test'], alpha=1.0, label='KL (Test)')
        ax2.set_ylim(0, max(self.train_df['kl_train']) + 10)
        ax2.set_ylabel('KL', color=color_dict['kl_train'])
        ax2.tick_params('y', colors=color_dict['kl_train'])

        # Gather lines and labels to create legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True, fancybox=True, shadow=False)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')

        # Add warmup line
        ax2.axvline(self.statistics['warm_up'], color='k', linewidth=0.5)
        xy = (float(self.statistics['warm_up'])/self.statistics['iter_train'][-1] + 0.1, 0.3)
        plt.annotate(r'$\tau = {}$'.format(self.statistics['warm_up']),
                     xy=xy,
                     xycoords="axes fraction",
                     horizontalalignment='center',
                     verticalalignment='center')

        # Align y-axis ticks to that grid doesn't get superimposed for KL/ELBO
        l1 = ax1.get_ylim()
        l2 = ax2.get_ylim()
        f = lambda x : l2[0]+(x-l1[0])/(l1[1]-l1[0])*(l2[1]-l2[0])
        ticks = f(ax1.get_yticks())
        ax2.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))

        # Set grid, only for first axis to not write over second axis
        ax1.grid(False)

        # Tight layout
        fig.tight_layout()

        # Save it all
        save_img_path = os.path.join(self.output_path, 'full_fig')
        fig.savefig(save_img_path + '.eps', dpi=400, format='eps')
        fig.savefig(save_img_path + '.png', dpi=400, format='png')
        print('Image saved to path {}.(eps/png)'.format(save_img_path))

        plt.close("all")

    def output_sentences(self):
        """Output all of the generated sentences"""
        npy_dict = {}
        generated_output = os.path.join(self.save_path, 'generated_output')
        # We first clean the directory of any .txt files
        [os.remove(os.path.join(generated_output, filename)) for filename in os.listdir(generated_output) if filename.endswith('.txt')]

        # Load all files
        for filename in os.listdir(generated_output):
            if filename.endswith('.npy') and not 'z' in filename.split('_'):
                npy_dict[filename.strip('.npy')] = np.load(os.path.join(generated_output, filename))

        sentence_dict = {}
        def if_en_else_fr_dict(key):
            if 'x' in key.split('_'):
                return self.word_dict_en
            else:
                return self.word_dict_fr

        # Make sentences from .npy files
        for key, value in npy_dict.items():
            sentence_dict[key] = translate_one_hot_to_words(value, if_en_else_fr_dict(key))

        for key in sentence_dict.keys():
            with codecs.open(os.path.join(generated_output, key + '.txt'), 'w', 'utf8') as f:
                f.write("\n".join(sentence_dict[key]))
                print('Wrote {}.txt to file'.format(key))

        print('All generated output written to disk')


def translate_one_hot_to_words(array, word_dict):
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


def plot_3_reconstruction(mlp_path, rnn_path, wavenet_path):
    """Plot the three plots with shared y-axis for reconstruction"""
    dict_path = 'fr_en_2to50_vocab30000_full'

    mlp_path = os.path.join(PATHS['save_dir'], mlp_path)
    rnn_path = os.path.join(PATHS['save_dir'], rnn_path)
    wavenet_path = os.path.join(PATHS['save_dir'], wavenet_path)

    mlp_er = EvaluateRunReconstruction(mlp_path, dict_path)
    rnn_er = EvaluateRunReconstruction(rnn_path, dict_path)
    wavenet_er = EvaluateRunReconstruction(wavenet_path, dict_path)

    # Set all styles
    plt.style.use('seaborn-dark')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 16

    color_dict = dict(elbo_train='#ef4747',
                      elbo_test='#ef7347',
                      kl_train='#47efc7',
                      kl_test='#47caef')

    # Create three axes which share x and y axes
    fig = plt.figure(figsize=(16, 6))
    axes = []
    for i in range(1, 3 + 1):
        ax = fig.add_subplot(1, 3, i)
        ax_twinx = ax.twinx()
        axes.append([ax, ax_twinx])

    # Calculate the ylimits
    ylim_kl_max = 100
    ylim_kl_min = 0
    ylim_elbo_max = -55
    ylim_elbo_min = -80

    tick_spacing = 5

    # Generate the first plots
    for i, run in enumerate([mlp_er, rnn_er, wavenet_er]):
        # Plot everything
        axes[i][0].plot(run.train_df['elbo_train'], color=color_dict['elbo_train'], alpha=1.0, label='ELBO (Train)')
        axes[i][0].plot(run.test_df['elbo_test'], color=color_dict['elbo_test'], alpha=1.0, label='ELBO (Test)')
        axes[i][0].set_ylim(ylim_elbo_min, ylim_elbo_max + 10)
        axes[i][1].plot(run.train_df['kl_train'], color=color_dict['kl_train'], alpha=1.0, label='KL (Train)')
        axes[i][1].plot(run.test_df['kl_test'], color=color_dict['kl_test'], alpha=1.0, label='KL (Test)')
        axes[i][1].set_ylim(0, ylim_kl_max + 10)

        # Add warmup line
        axes[i][1].axvline(run.statistics['warm_up'], color='k', linewidth=0.5)
        xy = (float(run.statistics['warm_up'])/run.statistics['iter_train'][-1] + 0.15, 0.9)
        axes[i][1].annotate(r'$\beta = {}$'.format(run.statistics['warm_up']),
                            xy=xy,
                            xycoords="axes fraction",
                            horizontalalignment='center',
                            verticalalignment='center')

        # Set grid, only for first axes to not write over second axes
        axes[i][0].grid(True)
        axes[i][1].grid(False)

    # Set titles
    axes[0][0].set_title('MLP')
    axes[1][0].set_title('RNN')
    axes[2][0].set_title('WaveNet')

    # Set xlabel in middle as iterations
    axes[1][0].set_xlabel('Iterations')

    # Remove unwanted axes by removing labels
    axes[0][1].set_yticklabels([])
    axes[1][0].set_yticklabels([])
    axes[1][1].set_yticklabels([])
    axes[2][0].set_yticklabels([])

    # Align y-axes ticks to that grid doesn't get superimposed for KL/ELBO
    l1 = axes[0][0].get_ylim()
    l2 = axes[0][1].get_ylim()
    f = lambda x : l2[0]+(x-l1[0])/(l1[1]-l1[0])*(l2[1]-l2[0])
    axes[0][0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    axes[0][0].set_ylabel('ELBO', color=color_dict['elbo_train'])
    axes[0][0].tick_params('y', colors=color_dict['elbo_train'])
    axes[0][0].spines['left'].set_visible(True)
    ticks = f(axes[0][0].get_yticks())
    axes[2][1].yaxis.set_major_locator(ticker.FixedLocator(ticks))
    axes[2][1].set_ylabel('KL', color=color_dict['kl_test'])
    axes[2][1].tick_params('y', colors=color_dict['kl_test'])
    axes[2][1].spines['right'].set_visible(True)

    # Gather lines and labels to create legend
    lines1, labels1 = axes[1][0].get_legend_handles_labels()
    lines2, labels2 = axes[1][1].get_legend_handles_labels()
    legend = fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=4, frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    # Tight layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    # Save it all
    save_img_path = os.path.join(PATHS['fig_dir'], 'reconstruction_compare_fig')
    fig.savefig(save_img_path + '.eps', dpi=400, format='eps')
    fig.savefig(save_img_path + '.png', dpi=400, format='png')
    print('Image saved to path {}.(eps/png)'.format(save_img_path))

    plt.close("all")

def generate_reconstruction_plots():
    """Generate side by side plots of reconstruction for rnn/mlp/wavenet"""
    os.chdir('./../models/')

    mlp_path = glob.glob('*mlp*reconstruction')
    rnn_path = glob.glob('*rnn*reconstruction')
    wavenet_path = glob.glob('*wavenet*reconstruction')


def generate_everything():
    """Generate all possible statistics for reconstruction and translation"""
    os.chdir('./../models/')

    save_paths_reconstruction = glob.glob('*reconstruction*')
    save_paths_translation = glob.glob('*translation*')

    print(save_paths_reconstruction)
    print(save_paths_translation)

    dict_path = 'fr_en_2to50_vocab30000_full'

    for save_path in save_paths_reconstruction:
        er = EvaluateRunReconstruction(save_path, dict_path)
        er.create_plots()
        er.output_sentences()

    for save_path in save_paths_translation:
        er = EvaluateRunTranslation(save_path, dict_path)
        er.create_plots()
        er.output_sentences()

    os.chdir('./../helper')

if __name__ == '__main__':
    #generate_everything()

    dict_path = 'fr_en_2to30_vocab30000_full'
    save_path = 'mlp_wavenet_translation_fr_en.20-08-17_18:16'
    er = EvaluateRunTranslation(save_path, dict_path)
    er.output_sentences()

    #generate_everything()

    # plot_3_reconstruction('mlp_reconstruction_output_lang_en_13-08-17_23:06',
    #                       'rnn_reconstruction_output_lang_en_14-08-17_12:12',
    #                       'wavenet_reconstruction_output_lang_en_15-08-17_06:44')
