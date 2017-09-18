"""
Generative models for translation project
"""

import sys

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import ConcatLayer, Gate, get_all_param_values, get_all_params, set_all_param_values, get_output, \
    GRULayer, InputLayer, DenseLayer, Conv1DLayer, ReshapeLayer, NonlinearityLayer, ElemwiseMergeLayer, ElemwiseSumLayer, DimshuffleLayer
from lasagne.nonlinearities import tanh, sigmoid, elu

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams()

sys.path.append('../models/')
sys.path.append('../data/')
sys.path.append('../external/')
sys.path.append("../nn")
sys.path.append('../tools/')
sys.path.append('../visualization/')
sys.path.append('../../')

from layers import DilatedConv1DLayer, RepeatLayer
from nonlinearities import elu_plus_one
from decoders import sample, viterbi

class GenAUTR(object):
    """Generative model class for AUTR"""

    def __init__(self,
                 max_length,
                 vocab_size,
                 num_time_steps,
                 nn_kwargs,
                 z_dim,
                 z_dist,
                 output_dist):
        """
        :param max_length: (int) maximum length of input data (x or y)
        :param vocab_size: (int) size of the vocabulary from input language
        :param num_time_steps: number of steps through the RNN
        :param nn_kwargs: (dict) dictionary of all of the needed kwargs for the
        neural network used by the generative model (same for x and y)
        :param z_dim: (int) latent dimensionality of z
        :param z_dist: (distribution) latent distribution (will define behaviour of this as well)
        NOTE: z_dim, z_dist have to be the same for recognition model
        and generative model by definition.
        :param output_dist: (distribution) output distribution
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.num_time_steps = num_time_steps

        # nn_kwargs consist of the following keys:
        # rnn_depth: depth of the RNN
        # hid_dim:
        # use_skip:
        # bidirectional:
        self.hid_depth = nn_kwargs['rnn_depth']
        self.hid_dim = nn_kwargs['hid_dim']
        self.use_skip = nn_kwargs['use_skip']
        self.bidirectional = nn_kwargs['bidirectional']

        self.z_dim = z_dim
        # z_dist is an instance of distance class
        self.z_dist = z_dist()
        self.dist_output = output_dist()

        self.rnn = self.rnn_fn()

        # define shared variables for all affine transformations: (W, b) for each Wx + b
        # the matrices are defined such that (input_dim, output_dim) is the dimensionality

        # if we are to use skip-connections in RNN
        if self.use_skip:
            self.W_h_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.hid_depth * self.hid_dim,
                                                                              self.max_length))))
            self.W_h_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.hid_depth * self.hid_dim,
                                                                              self.vocab_size * self.vocab_size))))
        else:
            self.W_h_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.hid_dim, self.max_length))))
            self.W_h_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.hid_dim,
                                                                              self.vocab_size * self.vocab_size))))

        self.W_Cg_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length,
                                                                           self.vocab_size * self.vocab_size))))
        self.b_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length,))))
        self.b_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.vocab_size * self.vocab_size,))))

        self.canvas_update_params = [self.W_h_Cg, self.b_Cg, self.W_h_Cu, self.W_Cg_Cu, self.b_Cu]

    def last_d_softmax(self, x):
        """Utility function for getting the transition table from canvas"""
        e_x = T.exp(x - x.max(axis=-1, keepdims=True))
        out = e_x / e_x.sum(axis=-1, keepdims=True)

        return out

    def rnn_fn(self):
        """Define the rnn using lasagne

        :return l_current: lasagne RNN"""
        l_in = InputLayer((None, None, self.z_dim))
        layers = [l_in]
        l_current = l_in

        # create the rnn layer
        for h in range(1, self.hid_depth + 1):
            backwards = True if self.bidirectional and h % 2 == 0 else False
            l_h = GRULayer(l_current, num_units=self.hid_dim, hidden_update=Gate(nonlinearity=tanh),
                           backwards=backwards)

            # if we want to use skip-connections we concatenate the current layer
            if self.use_skip:
                layers.append(l_h)
                if h != self.hid_depth:
                    l_current = ConcatLayer([l_in, l_h], axis=2)
                else:
                    l_current = ConcatLayer(layers[1:], axis=2)
            else:
                l_current = l_h

        return l_current

    def canvas_updater(self, hiddens):
        """Update the canvas

        :param hiddens: N * T * dim(hid) tensor

        :return canvases: T * max(L) * D * D tensor
        :return gate_sum_init: T * max(L)
        """
        canvas_init = T.zeros((hiddens.shape[0], self.max_length, self.vocab_size, self.vocab_size))
        gate_sum_init = T.zeros((hiddens.shape[0], self.max_length))

        def step(h_t, canvas_tm1, canvas_gate_sum_tm1, W_h_Cg, b_Cg, W_h_Cu, W_Cg_Cu, b_Cu):
            """Calculate the canvas update step"""
            pre_softmax_gate = T.dot(h_t, W_h_Cg) + b_Cg.reshape((1, b_Cg.shape[0]))  # N * max(L)

            gate_exp = T.exp(pre_softmax_gate - pre_softmax_gate.max(axis=-1, keepdims=True))
            unnormalised_gate = gate_exp * (T.ones_like(canvas_gate_sum_tm1) - canvas_gate_sum_tm1)  # N * max(L)
            canvas_gate = unnormalised_gate / unnormalised_gate.sum(axis=-1, keepdims=True)  # N * max(L)
            canvas_gate *= (T.ones_like(canvas_gate_sum_tm1) - canvas_gate_sum_tm1)  # N * max(L)

            canvas_gate_sum = canvas_gate_sum_tm1 + canvas_gate  # (S*N) * max(L) matrix
            canvas_gate_reshape = canvas_gate.reshape((canvas_gate.shape[0], canvas_gate.shape[1], 1, 1))  # N * max(L)
            # * 1 * 1

            canvas_update = T.dot(h_t, W_h_Cu) + T.dot(canvas_gate, W_Cg_Cu) + b_Cu.reshape((1, b_Cu.shape[0]))  # N *
            # (D*D)
            canvas_update = canvas_update.reshape((canvas_update.shape[0], 1, self.vocab_size, self.vocab_size))  # N *
            # 1 * D * D

            canvas_new = ((T.ones_like(canvas_gate_reshape) - canvas_gate_reshape) * canvas_tm1) + \
                         (canvas_gate_reshape * canvas_update)  # N * max(L) * D * D

            return T.cast(canvas_new, 'float32'), T.cast(canvas_gate_sum, 'float32')

        ([canvases, canvas_gate_sums], _) = theano.scan(step,
                                                        sequences=[hiddens.dimshuffle((1, 0, 2))],
                                                        outputs_info=[canvas_init, gate_sum_init],
                                                        non_sequences=self.canvas_update_params,
                                                        )

        return canvases[-1], canvas_gate_sums[-1]

    def get_canvases(self, z, num_time_steps=None):
        """
        :param z: N * dim(z) matrix
        :param num_time_steps: int - number of RNN time steps to use

        :return canvases: N * max(L) * D * D tensor
        :return canvas_gate_sums: N * max(L) matrix
        """
        if num_time_steps is None:
            num_time_steps = self.num_time_steps

        z_rep = T.tile(z.reshape((z.shape[0], 1, z.shape[1])), (1, num_time_steps, 1))  # N * T * dim(z)
        hiddens = get_output(self.rnn, z_rep)  # N * T * dim(hid)
        canvases, canvas_gate_sums = self.canvas_updater(hiddens)  # N * max(L) * D * D and N * max(L)

        return canvases, canvas_gate_sums

    def get_trans_probs(self, z):
        """
        :param z: N * dim(z) matrix

        :return trans_probs: N * max(L) * D * D tensor
        """
        canvas = self.get_canvases(z)[0]  # N * max(L) * D * D
        trans_probs = self.last_d_softmax(canvas)  # N * max(L) * D * D

        return trans_probs

    def log_p_z(self, z):
        """
        :param z: (S*N) * dim(z) matrix

        :return log_p_z: (S*N) length vector
        """
        log_p_z = self.z_dist.log_density(z)

        return log_p_z

    def compute_log_p_x(self, trans_probs, rep):
        """
        :param trans_probs: (S*N) * max(L) * D * D tensor
        :param y_rep: (S*N) * max(L) * D tensor

        :return: log_p_y: (S*N) length vector
        """

        zeros_init = T.zeros((rep.shape[0], 1, self.vocab_size))  # (S*N) * 1 * D

        pre_padded_rep = T.concatenate([zeros_init, rep], axis=1)[:, :-1, :]  # (S*N) * max(L) * D
        rep_inds = T.argmax(pre_padded_rep, axis=-1)  # (S*N) * max(L) matrix

        trans_probs_reshape = trans_probs.reshape((-1, self.vocab_size, self.vocab_size))  # ((S*N)*max(L)) * D * D
        probs_flat = trans_probs_reshape[T.arange(T.prod(rep_inds.shape)), rep_inds.flatten()]
        probs = probs_flat.reshape((rep.shape[0], self.max_length, self.vocab_size))  # (S*N) * max(L) * D tensor=
        log_p_x = self.dist_output.log_density(rep, [probs])  # (S*N)

        return log_p_x

    def log_p_x(self, x, z):
        """
        :param y: N * max(L) * D tensor
        :param z: (S*N) * dim(z) matrix

        :return log_p_x: (S*N) length vector
        """
        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        rep = T.tile(x, (S, 1, 1))  # (S*N) * max(L) * D
        canvas = self.get_canvases(z)[0]  # (S*N) * max(L) * D * D
        trans_probs = self.last_d_softmax(canvas)  # (S*N) * max(L) * D * D

        return self.compute_log_p_x(trans_probs, rep)  # (S*N)

    def generate_output_prior_fn(self, num_samples, num_samples_per_sample, only_final):
        """
        :param num_samples: number of samples
        :param num_samples_per_sample: number to average the samples over
        :param only_final: only pick the final output

        :return generate_output_prior_fn: theano function"""
        # get canvas and transition tensor from samples latents
        z = self.z_dist.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)
        canvas, canvas_gate_sums = self.get_canvases(z)  # S * max(L) * D * D tensor and S * max(L) matrix
        trans_probs_output = self.last_d_softmax(canvas)  # S * max(L) * D * D tensor

        if only_final:
            chars_viterbi, probs_viterbi = viterbi(trans_probs_output)  # S * max(L) matrix
        else:
            chars_viterbi = []
            probs_viterbi = []
            canvas_gate_sums = []

            # generate the most probably sequence per timestep t using viterbi
            for t in range(1, self.num_time_steps + 1):
                canvas_t, canvas_gate_sums_t = self.get_canvases(z, num_time_steps=t)  # S * max(L) * D * D tensor and
                # S * max(L) matrix

                # generate most probably sequence
                trans_probs_t = self.last_d_softmax(canvas_t)  # S * max(L) * D * D tensor
                chars_viterbi_t, probs_viterbi_t = viterbi(trans_probs_t)  # S * max(L) matrix

                # append this sequence to list
                chars_viterbi.append(chars_viterbi_t)
                probs_viterbi.append(probs_viterbi_t)

                # append current canvas gate sums to list
                canvas_gate_sums.append(canvas_gate_sums_t)

            chars_viterbi = T.stack(chars_viterbi, axis=0)  # T * S * max(L) tensor
            probs_viterbi = T.stack(probs_viterbi, axis=0)  # T * S * max(L) tensor
            canvas_gate_sums = T.stack(canvas_gate_sums, axis=0)  # T * S * max(L) tensor

        chars_sampled, probs_sampled, updates = sample(trans_probs_output, num_samples_per_sample)  # S*SpS * max(L) matrix

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, trans_probs_output, canvas_gate_sums, chars_viterbi,
                                                         probs_viterbi, chars_sampled],
                                                updates=updates,
                                                on_unused_input='ignore',
                                                )

        return generate_output_prior

    def generate_output_posterior_fn(self, x, y, z, num_samples_per_sample, only_final):
        """
        :param z: latent variables
        :param num_samples_per_sample: number to average the samples over
        :param only_final: only pick the final output

        :return generate_output_posterior_fn: theano function"""
        # get canvas and transition tensor from samples latents
        canvas, canvas_gate_sums = self.get_canvases(z)  # N * max(L) * D * D tensor and N * max(L) matrix
        trans_probs = self.last_d_softmax(canvas)

        if only_final:
            chars_viterbi, probs_viterbi = viterbi(trans_probs)  # N * max(L) matrix
        else:
            chars_viterbi = []
            probs_viterbi = []
            canvas_gate_sums = []

            # generate the most probable sequence per timestep t using viterbi
            for t in range(1, self.num_time_steps + 1):
                canvas_t, canvas_gate_sums_t = self.get_canvases(z, num_time_steps=t)  # S * max(L) * D * D tensor and
                # S * max(L) matrix

                # generate most probably sequence
                trans_probs_t = self.last_d_softmax(canvas_t)  # S * max(L) * D * D tensor
                chars_viterbi_t, probs_viterbi_t = viterbi(trans_probs_t)  # S * max(L) matrix

                # append this sequence to list
                chars_viterbi.append(chars_viterbi_t)
                probs_viterbi.append(probs_viterbi_t)

                # append current canvas gate sums to list
                canvas_gate_sums.append(canvas_gate_sums_t)

            chars_viterbi = T.stack(chars_viterbi, axis=0)  # T * S * max(L) tensor
            probs_viterbi = T.stack(probs_viterbi, axis=0)  # T * S * max(L) tensor
            canvas_gate_sums = T.stack(canvas_gate_sums, axis=0)  # T * S * max(L) tensor

        chars_sampled, probs_sampled, updates = sample(trans_probs, num_samples_per_sample)  # N*SpS * max(L) matrix

        generate_output_posterior = theano.function(inputs=[x, y],
                                                    outputs=[z, trans_probs, canvas_gate_sums, chars_viterbi,
                                                             probs_viterbi, chars_sampled],
                                                    updates=updates,
                                                    allow_input_downcast=True,
                                                    on_unused_input='ignore',
                                                    )

        return generate_output_posterior

    def follow_latent_trajectory_fn(self, alphas, num_samples):
        """Produce the homotopy between z1 and z2
        :param alphas: linspace between 0 and 1 to produce homotopies
        (here linear transformation  z = z1 * t + (1 - t) * z2)
        :param num_samples: number of samples to generate

        :return follow_latent_trajectory: theano function"""
        # sample latents
        z1 = self.z_dist.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)
        z2 = self.z_dist.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)

        # tile in order to make it consistent with alphas
        z1_rep = T.extra_ops.repeat(z1, alphas.shape[0], axis=0)  # (S*A) * dim(z)
        z2_rep = T.extra_ops.repeat(z2, alphas.shape[0], axis=0)  # (S*A) * dim(z)

        alphas_rep = T.tile(alphas, num_samples)  # (S*A)

        z = (T.shape_padright(alphas_rep) * z1_rep) + (T.shape_padright(T.ones_like(alphas_rep) - alphas_rep) * z2_rep)
        # (S*A) * dim(z)

        # get canvas for this latent z and run viterbi
        canvas, canvas_gate_sums = self.get_canvases(z)  # S * max(L) * D * D and S * max(L)
        trans_probs_y = self.last_d_softmax(canvas)  # S * max(L) * D * D
        chars_viterbi, probs_viterbi = viterbi(trans_probs_y)  # S * max(L)

        follow_latent_trajectory = theano.function(inputs=[alphas],
                                                   outputs=[chars_viterbi, probs_viterbi],
                                                   allow_input_downcast=True
                                                   )

        return follow_latent_trajectory

    def get_params(self):
        """Get all parameters of the model"""
        rnn_params = get_all_params(self.rnn, trainable=True)

        return rnn_params + self.canvas_update_params

    def get_param_values(self):
        """Get the parameters of the model, non-mutable"""
        rnn_params_vals = get_all_param_values(self.rnn)
        canvas_update_params_vals = [p.get_value() for p in self.canvas_update_params]

        return [rnn_params_vals, canvas_update_params_vals]

    def set_param_values(self, param_values):
        """Set the parameters of the model"""
        [rnn_params_vals, canvas_update_params_vals] = param_values
        set_all_param_values(self.rnn, rnn_params_vals)

        for i in range(len(self.canvas_update_params)):
            self.canvas_update_params[i].set_value(canvas_update_params_vals[i])

class GenWaveNetTextWords(object):
    """Generative model class for AUTR"""

    def __init__(self, max_length, vocab_size, nn_kwargs, z_dim, embedding_dim, embedder, z_dist, output_dist):
        """
        :param max_length: (int) maximum length of input data (x or y)
        :param vocab_size: (int) size of the vocabulary from input language
        :param nn_kwargs: (dict) dictionary of all of the needed kwargs for the
        neural network used by the generative model (same for x and y)
        :param z_dim: (int) latent dimensionality of z
        :param embedding_dim: (int) embedding dimensionality
        :param z_dist: (distribution) latent distribution (will define behaviour of this as well)
        NOTE: z_dim, z_dist have to be the same for recognition model
        and generative model by definition.
        :param output_dist: (distribution) output distribution
        """
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.embedder = embedder

        self.nn_dilations = nn_kwargs['dilations']
        self.nn_dilation_channels = nn_kwargs['dilation_channels']
        self.nn_residual_channels = nn_kwargs['residual_channels']

        self.z_dim = z_dim
        self.embedding_dim = embedding_dim
        # z_dist is an instance of distance class
        self.z_dist = z_dist()
        self.output_dist = output_dist()

        self.nn_input_layers, self.nn_output_layer = self.nn_fn()

    def nn_fn(self):
        l_in_x = InputLayer((None, self.embedding_dim, self.max_length))
        l_in_z = InputLayer((None, self.z_dim))
        l_causal_conv = DilatedConv1DLayer(l_in_x, num_filters=self.nn_residual_channels, dilation=1, nonlinearity=None)
        l_prev = l_causal_conv

        skip_layers = []

        for h in range(len(self.nn_dilations)):
            l_x_filter = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels,
                                            dilation=self.nn_dilations[h], nonlinearity=None, b=None)

            l_z_filter = DenseLayer(l_in_z, num_units=self.nn_dilation_channels, nonlinearity=None, b=None)
            l_z_filter_reshape = ReshapeLayer(l_z_filter, ([0], [1], 1,))
            l_z_filter_rep = RepeatLayer(l_z_filter_reshape, self.max_length, axis=-1, ndim=3)

            l_filter = NonlinearityLayer(ElemwiseSumLayer([l_x_filter, l_z_filter_rep]), nonlinearity=tanh)

            l_x_gate = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels, dilation=self.nn_dilations[h],
                                          nonlinearity=None, b=None)

            l_z_gate = DenseLayer(l_in_z, num_units=self.nn_dilation_channels, nonlinearity=None, b=None)
            l_z_gate_reshape = ReshapeLayer(l_z_gate, ([0], [1], 1,))
            l_z_gate_rep = RepeatLayer(l_z_gate_reshape, self.max_length, axis=-1, ndim=3)

            l_gate = NonlinearityLayer(ElemwiseSumLayer([l_x_gate, l_z_gate_rep]), nonlinearity=sigmoid)

            l_merge = ElemwiseMergeLayer([l_filter, l_gate], merge_function=T.mul)

            l_dense = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None, b=None)

            l_residual = ElemwiseSumLayer([l_prev, l_dense])

            l_skip = Conv1DLayer(l_merge, num_filters=self.embedding_dim, filter_size=1, nonlinearity=None, b=None)

            skip_layers.append(l_skip)

            l_prev = l_residual

        l_skip_sum = NonlinearityLayer(ElemwiseSumLayer(skip_layers), nonlinearity=elu)
        l_prev = l_skip_sum

        for h in range(2):
            l_h = Conv1DLayer(l_prev, num_filters=self.embedding_dim, filter_size=1, nonlinearity=None, b=None)
            l_z = DenseLayer(l_in_z, num_units=self.embedding_dim, nonlinearity=None, b=None)
            l_z_reshape = ReshapeLayer(l_z, ([0], [1], 1,))
            l_z_reshape_rep = RepeatLayer(l_z_reshape, self.max_length, axis=-1, ndim=3)
            l_sum = NonlinearityLayer(ElemwiseSumLayer([l_h, l_z_reshape_rep]), nonlinearity=elu)
            l_prev = l_sum

        l_out = DimshuffleLayer(l_prev, (0, 2, 1))

        return (l_in_x, l_in_z), l_out

    def last_d_softmax(self, x):
        """Utility function for getting the transition table from canvas"""
        e_x = T.exp(x - x.max(axis=-1, keepdims=True))
        out = e_x / e_x.sum(axis=-1, keepdims=True)

        return out

    def get_probs(self, x, z, all_embeddings, approximate_by_css=False, css_num_samples=None, css_probs=None, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param all_embeddings: D * E matrix
        :param approximate_by_css: bool
        :param css_samples: css_num_samples length vector
        :param css_is_weights: css_num_samples length vector
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor
        """

        SN = z.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        hiddens = get_output(self.nn_output_layer, {self.nn_input_layers[0]: x_pre_padded.dimshuffle((0, 2, 1)),
                                                    self.nn_input_layers[1]: z})
        # (S*N) * max(L) * E

        if approximate_by_css:
            css_samples = random.choice(p=T.shape_padleft(css_probs), size=css_num_samples, replace=False).flatten()
            # css_num_samples

            css_samples_embedded = T.shape_padleft(all_embeddings[css_samples], n_ones=2)  # 1 * 1 * css_num_samples * E

            css_samples_embedded_rep = T.tile(css_samples_embedded, (SN, self.max_length, 1, 1))  # (S*N) * max(L) *
            # css_num_samples * E

            css_is_weights = T.cast(1., 'float32') / (css_probs[css_samples] * css_num_samples)  # css_num_samples

            probs_numerators = T.exp(T.sum(x * hiddens, axis=-1))  # (S*N) * max(L)

            probs_denominators = T.sum(T.exp(T.sum(css_samples_embedded_rep * T.shape_padaxis(hiddens, 2), axis=-1))
                                       * T.shape_padleft(css_is_weights, n_ones=2), axis=-1) + probs_numerators
            # (S*N) * max(L)

            probs = probs_numerators / probs_denominators

        else:

            probs_numerators = T.sum(x * hiddens, axis=-1)  # (S*N) * max(L)

            probs_denominators = T.dot(hiddens, all_embeddings.T)  # (S*N) * max(L) * D

            if mode == 'all':
                probs = self.last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
            elif mode == 'true':
                probs_numerators -= T.max(probs_denominators, axis=-1)
                probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

                probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
            else:
                raise Exception("mode must be in ['all', 'true']")

        return probs

    def log_p_z(self, z):
        """
        :param z: (S*N) * dim(z) matrix

        :return log_p_z: (S*N) length vector
        """
        log_p_z = self.z_dist.log_density(z)

        return log_p_z

    def log_p_x(self, x, z, all_embeddings, approximate_by_css, css_num_samples, vocab_counts):
        """
        :param x: N * max(L) tensor
        :param z: (S*N) * dim(z) matrix
        :param all_embeddings: D * E matrix
        :param approximate_by_css: bool
        :param css_samples: css_num_samples length vector
        :param css_is_weights: css_num_samples length vector

        :return log_p_x: (S*N) length vector
        """

        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        x_rep = T.tile(x, (S, 1))  # (S*N) * max(L)

        if approximate_by_css:

            sampling_mask = T.ones((self.vocab_size,))  # D
            sampling_mask = T.set_subtensor(sampling_mask[x.flatten()], 0)  # D

            vocab_counts_exc_true = vocab_counts * sampling_mask  # D

            css_probs = vocab_counts_exc_true / T.sum(vocab_counts_exc_true, keepdims=True)  # D

        else:

            css_probs = None

        x_rep_padding_mask = T.switch(T.lt(x_rep, 0), 0, 1)  # (S*N) * max(L)

        x_rep_embedded = self.embedder(x_rep, all_embeddings)  # (S*N) * max(L) * E

        probs = self.get_probs(x_rep_embedded, z, all_embeddings, approximate_by_css, css_num_samples, css_probs,
                               mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-15, 'float32')  # (S*N) * max(L)

        log_p_x = T.sum(x_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        return log_p_x

    def generate_text(self, z, all_embeddings):
        """
        :param z: N * dim(z) matrix
        :param all_embeddings: D * E matrix

        :return x: N * max(L) tensor
        """

        N = z.shape[0]

        x_init_sampled = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)
        x_init_argmax = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)

        def step(l, x_prev_sampled, x_prev_argmax, z, all_embeddings):

            x_prev_sampled_embedded = self.embedder(x_prev_sampled, all_embeddings)  # N * max(L) * E

            probs_sampled = self.get_probs(x_prev_sampled_embedded, z, all_embeddings, mode='all')  # N * max(L) * D

            x_sampled_one_hot = self.output_dist.get_samples([T.shape_padaxis(probs_sampled[:, l], 1)])  # N * 1 * D

            x_sampled_l = T.argmax(x_sampled_one_hot, axis=-1).flatten()  # N

            x_current_sampled = T.set_subtensor(x_prev_sampled[:, l], x_sampled_l)  # N * max(L)

            #

            x_prev_argmax_embedded = self.embedder(x_prev_argmax, all_embeddings)  # N * max(L) * E

            probs_argmax = self.get_probs(x_prev_argmax_embedded, z, all_embeddings, mode='all')  # N * max(L) * D

            x_argmax_l = T.argmax(probs_argmax[:, l], axis=-1)  # N

            x_current_argmax = T.set_subtensor(x_prev_argmax[:, l], x_argmax_l)  # N * max(L)

            return T.cast(x_current_sampled, 'int32'), T.cast(x_current_argmax, 'int32')

        (x_sampled, x_argmax), updates = theano.scan(step,
                                                     sequences=[T.arange(self.max_length)],
                                                     outputs_info=[x_init_sampled, x_init_argmax],
                                                     non_sequences=[z, all_embeddings],
                                                     )

        return x_sampled[-1], x_argmax[-1], updates

    def beam_search(self, z, all_embeddings, beam_size=5):
        """
        :param z: N * dim(z) matrix
        :param all_embeddings: D * E matrix
        :param beam_size: B int

        :return x: N * max(L) tensor
        """

        N = z.shape[0]

        # Repeat so that each beam gets the same for each N
        z_start = T.repeat(z, beam_size, axis=0)  # (N*B) * E

        # Initialise the scores, edges and words
        best_scores_0 = T.zeros((N, beam_size), 'float32')  # N * B
        best_edges_0 = T.zeros((N, self.vocab_size), 'int32')  # N * D
        active_words_init = T.cast(-1, 'int32') * T.ones((N, beam_size, self.max_length), 'int32')  # N * B * max(L)

        def step_forward(l, best_scores_lm1, best_edges_lm1, active_words_current, z_start, all_embeddings):
            # Go from the normal word space to the embedded space and reshape into a form
            # where we get rid of the extra 'B' axis
            active_words_embedded = self.embedder(active_words_current, all_embeddings)  # N * B * max(L) * E
            active_words_embedded_reshape = T.reshape(active_words_embedded, (N*beam_size, self.max_length, self.embedding_dim))  # (N*B) * max(L) * E

            # Get the probability at word l and calculate the scores by combining previously
            # calculated scores
            probs = self.get_probs(active_words_embedded_reshape, z_start, all_embeddings, mode='all')[:, l]  # (N*B) * D
            probs_reshaped = probs.reshape((N, beam_size, self.vocab_size))  # N * B * D
            scores = T.shape_padright(best_scores_lm1) + T.log(probs_reshaped)  # N * B * D

            # Get best scores and prune the tree
            best_scores_l_all = T.max(scores, axis=1)  # N * D
            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            # Get best words that led to best scores
            active_words_l = T.argsort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B
            active_words_new = T.set_subtensor(active_words_current[:, :, l], active_words_l)  # N * B * max(L)

            # Get best edges that led to best scores
            best_edges_l_beam_inds = T.argmax(scores, axis=1)  # N * D
            best_edges_l = active_words_current[:, :, l-1][T.repeat(T.arange(N), self.vocab_size), best_edges_l_beam_inds.flatten()]
            best_edges_l = best_edges_l.reshape((N, self.vocab_size))  # N * D

            # best_scores_l = T.zeros((N, beam_size), 'float32')
            # best_edges_l = T.zeros((N, self.vocab_size), 'int32')
            # active_words_new = T.ones((N, beam_size, self.max_length), 'int32')

            return T.cast(best_scores_l, 'float32'), T.cast(best_edges_l, 'int32'), T.cast(active_words_new, 'int32')

        (best_scores, best_edges, active_words), updates = theano.scan(step_forward,
                                                                       sequences=T.arange(self.max_length),
                                                                       outputs_info=[best_scores_0, best_edges_0, active_words_init],
                                                                       non_sequences=[z_start, all_embeddings])
        # max(L) * N * B and max(L) * N * D and max(L) * N * B * max(L)

        # Only recover the best one
        words_L = active_words[-1, :, -1, -1]  # N

        # Step backward to recover the best trace
        def step_backward(best_edges_l, words_lp1):
            words_l = best_edges_l[T.arange(N), T.cast(words_lp1, 'int32')]  # N

            return words_l

        words, _ = theano.scan(step_backward,
                               sequences=best_edges[1:][::-1],
                               outputs_info=[words_L],
                               )

        words = words[::-1]  # (max(L)-1) * N
        words = T.concatenate([words, T.shape_padleft(words_L)], axis=0).T  # N * max(L)

        return T.cast(words, 'int32'), updates

    def generate_output_prior_fn(self, all_embeddings, num_samples, beam_size):
        z = self.z_dist.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)
        x_gen, updates = self.beam_search(z, all_embeddings, beam_size)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen],
                                                updates=updates,
                                                allow_input_downcast=True,
                                                on_unused_input='ignore')

        return generate_output_prior

    def generate_output_posterior_fn(self, x, y, z, all_embeddings, beam_size):
        x_gen, updates = self.beam_search(z, all_embeddings, beam_size)

        generate_output_posterior = theano.function(inputs=[x, y],
                                                    outputs=[z, x_gen],
                                                    updates=updates,
                                                    allow_input_downcast=True,
                                                    on_unused_input='ignore',
                                                    )

        return generate_output_posterior

    def get_params(self):
        """Get all parameters of the model"""
        nn_params = get_all_params(self.nn_output_layer, trainable=True)
        return nn_params

    def get_param_values(self):
        """Get the parameters of the model, non-mutable"""
        nn_params_vals = get_all_param_values(self.nn_output_layer)
        return [nn_params_vals]

    def set_param_values(self, param_values):
        """Set the parameters of the model"""
        [nn_params_vals] = param_values
        set_all_param_values(self.nn_output_layer, nn_params_vals)


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

def last_d_softmax(x):

    e_x = T.exp(x - x.max(axis=-1, keepdims=True))
    out = e_x / e_x.sum(axis=-1, keepdims=True)

    return out
