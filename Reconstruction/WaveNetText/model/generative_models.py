import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import ConcatLayer, Conv1DLayer, DenseLayer, DimshuffleLayer, ElemwiseMergeLayer, ElemwiseSumLayer,\
    get_all_param_values, get_all_params, get_output, InputLayer, NonlinearityLayer, ReshapeLayer, set_all_param_values
from lasagne.nonlinearities import elu, linear, sigmoid, tanh
from nn.layers import DilatedConv1DLayer, RepeatLayer

from .utilities import last_d_softmax

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams()


class GenWaveNetText(object):

    def __init__(self, z_dim, max_length, vocab_size, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.nn_dilations = nn_kwargs['dilations']
        self.nn_dilation_channels = nn_kwargs['dilation_channels']
        self.nn_residual_channels = nn_kwargs['residual_channels']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.nn_input_layers, self.nn_output_layer = self.nn_fn()

    def nn_fn(self):

        l_in_x = InputLayer((None, self.vocab_size, self.max_length))

        l_in_z = InputLayer((None, self.z_dim))

        l_causal_conv = DilatedConv1DLayer(l_in_x, num_filters=self.nn_residual_channels, dilation=1, nonlinearity=None)

        l_prev = l_causal_conv

        skip_layers = []

        for h in range(len(self.nn_dilations)):

            l_x_filter = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels,
                                            dilation=self.nn_dilations[h], nonlinearity=None)

            l_z_filter = DenseLayer(l_in_z, num_units=self.nn_dilation_channels, nonlinearity=None)
            l_z_filter_reshape = ReshapeLayer(l_z_filter, ([0], [1], 1,))
            l_z_filter_rep = RepeatLayer(l_z_filter_reshape, self.max_length, axis=-1, ndim=3)

            l_filter = NonlinearityLayer(ElemwiseSumLayer([l_x_filter, l_z_filter_rep]), nonlinearity=tanh)

            l_x_gate = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels, dilation=self.nn_dilations[h],
                                          nonlinearity=None)

            l_z_gate = DenseLayer(l_in_z, num_units=self.nn_dilation_channels, nonlinearity=None)
            l_z_gate_reshape = ReshapeLayer(l_z_gate, ([0], [1], 1,))
            l_z_gate_rep = RepeatLayer(l_z_gate_reshape, self.max_length, axis=-1, ndim=3)

            l_gate = NonlinearityLayer(ElemwiseSumLayer([l_x_gate, l_z_gate_rep]), nonlinearity=sigmoid)

            l_merge = ElemwiseMergeLayer([l_filter, l_gate], merge_function=T.mul)

            l_dense = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None)

            l_residual = ElemwiseSumLayer([l_prev, l_dense])

            l_skip = Conv1DLayer(l_merge, num_filters=self.vocab_size, filter_size=1, nonlinearity=None)

            skip_layers.append(l_skip)

            l_prev = l_residual

        l_skip_sum = NonlinearityLayer(ElemwiseSumLayer(skip_layers), nonlinearity=elu)

        l_out_1 = Conv1DLayer(l_skip_sum, num_filters=self.vocab_size, filter_size=1, nonlinearity=elu)

        l_out_2 = Conv1DLayer(l_out_1, num_filters=self.vocab_size, filter_size=1, nonlinearity=None)

        l_out = DimshuffleLayer(l_out_2, (0, 2, 1))

        return (l_in_x, l_in_z), l_out

    def log_p_z(self, z):
        """
        :param z: (S*N) * dim(z) matrix

        :return log_p_z: (S*N) length vector
        """

        log_p_z = self.dist_z.log_density(z)

        return log_p_z

    def get_probs(self, x, z):
        """
        :param x: (S*N) * max(L) * D tensor
        :param z: (S*N) * dim(z) array

        :return probs: (S*N) * max(L) * D tensor
        """

        x_pre_padded = T.concatenate([T.zeros((x.shape[0], 1, self.vocab_size)), x], axis=1)[:, :-1]  # (S*N) * max(L) *
        # D

        hiddens = get_output(self.nn_output_layer, {self.nn_input_layers[0]: x_pre_padded.dimshuffle((0, 2, 1)),
                                                    self.nn_input_layers[1]: z})
        # (S*N) * max(L) * D

        probs = last_d_softmax(hiddens)  # (S*N) * max(L) * D

        return probs

    def log_p_x(self, x, z):
        """
        :param x: N * max(L) * D tensor
        :param z: (S*N) * dim(z) matrix

        :return log_p_x: (S*N) length vector
        """

        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        x_rep = T.tile(x, (S, 1, 1))  # (S*N) * max(L) * D

        probs = self.get_probs(x_rep, z)  # (S*N) * max(L) * D

        log_p_x = self.dist_x.log_density(x_rep, [probs])  # (S*N)

        return log_p_x

    def generate_text(self, z):
        """
        :param z: N * dim(z) matrix

        :return x: N * max(L) tensor
        """

        x_init_sampled = T.zeros((z.shape[0], self.max_length, self.vocab_size))  # N * max(L) * D
        x_init_argmax = T.zeros((z.shape[0], self.max_length, self.vocab_size))  # N * max(L) * D

        def step(l, x_prev_sampled, x_prev_argmax, z):

            probs_sampled = self.get_probs(x_prev_sampled, z)  # N * max(L) * D

            samples = self.dist_x.get_samples([T.shape_padaxis(probs_sampled[:, l], 1)])  # N * 1 * D

            x_current_sampled = T.set_subtensor(x_prev_sampled[:, l],
                                                samples.reshape((samples.shape[0], self.vocab_size)))  # N * max(L) * D

            probs_argmax = self.get_probs(x_prev_argmax, z)  # N * max(L) * D

            x_argmax = T.argmax(probs_argmax, axis=-1).flatten()  # (N*max(L))

            x_argmax_one_hot = T.zeros((x_argmax.shape[0], self.vocab_size))
            x_argmax_one_hot = T.set_subtensor(x_argmax_one_hot[T.arange(x_argmax.shape[0]), x_argmax], 1)
            x_argmax_one_hot = x_argmax_one_hot.reshape((probs_argmax.shape[0], self.max_length, self.vocab_size))
            # N * max(L) * D

            x_current_argmax = T.set_subtensor(x_prev_argmax[:, l], x_argmax_one_hot[:, l])  # N * max(L) * D

            return T.cast(x_current_sampled, 'float32'), T.cast(x_current_argmax, 'float32')

        (x_sampled, x_argmax), updates = theano.scan(step,
                                                     sequences=[T.arange(self.max_length)],
                                                     outputs_info=[x_init_sampled, x_init_argmax],
                                                     non_sequences=[z],
                                                     )

        return T.argmax(x_sampled[-1], axis=-1), T.argmax(x_argmax[-1], axis=-1), updates

    def generate_output_prior_fn(self, num_samples):

        z = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen_sampled, x_gen_argmax],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_output_prior

    def generate_output_posterior_fn(self, x, z):

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z)

        generate_output_posterior = theano.function(inputs=[x],
                                                    outputs=[z, x_gen_sampled, x_gen_argmax],
                                                    updates=updates,
                                                    allow_input_downcast=True
                                                    )

        return generate_output_posterior

    # def follow_latent_trajectory_fn(self, alphas, num_samples):
    #
    #     z1 = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)
    #     z2 = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)
    #
    #     z1_rep = T.extra_ops.repeat(z1, alphas.shape[0], axis=0)  # (S*A) * dim(z)
    #     z2_rep = T.extra_ops.repeat(z2, alphas.shape[0], axis=0)  # (S*A) * dim(z)
    #
    #     alphas_rep = T.tile(alphas, num_samples)  # (S*A)
    #
    #     z = (T.shape_padright(alphas_rep) * z1_rep) + (T.shape_padright(T.ones_like(alphas_rep) - alphas_rep) * z2_rep)
    #     # (S*A) * dim(z)
    #
    #     canvas, canvas_gate_sums = self.get_canvases(z)  # S * max(L) * D * D and S * max(L)
    #
    #     trans_probs_x = last_d_softmax(canvas)  # S * max(L) * D * D
    #
    #     chars_viterbi, probs_viterbi = viterbi(trans_probs_x)  # S * max(L)
    #
    #     follow_latent_trajectory = theano.function(inputs=[alphas],
    #                                                outputs=[chars_viterbi, probs_viterbi],
    #                                                allow_input_downcast=True
    #                                                )
    #
    #     return follow_latent_trajectory
    #
    # def find_best_matches_fn(self, sentences_orig, sentences_one_hot, batch_orig, batch_one_hot, z):
    #     """
    #     :param sentences_one_hot: S * max(L) X D tensor
    #     :param batch_one_hot: N * max(L) X D tensor
    #     :param z: S * dim(z) matrix
    #     """
    #
    #     S = sentences_one_hot.shape[0]
    #     N = batch_one_hot.shape[0]
    #
    #     canvas = self.get_canvases(z)[0]  # S * max(L) * D * D tensor
    #
    #     trans_probs = last_d_softmax(canvas)  # S * max(L) * D * D tensor
    #     trans_probs_rep = T.extra_ops.repeat(trans_probs, N, axis=0)  # (S*N) * max(L) * D * D tensor
    #
    #     batch_rep = T.tile(batch_one_hot, (S, 1, 1))  # (S*N) * max(L) * D tensor
    #
    #     log_p_batch = self.compute_log_p_x(trans_probs_rep, batch_rep)
    #
    #     log_p_batch = log_p_batch.reshape((S, N))
    #
    #     find_best_matches = theano.function(inputs=[sentences_orig, batch_orig],
    #                                         outputs=log_p_batch,
    #                                         allow_input_downcast=True,
    #                                         on_unused_input='ignore',
    #                                         )
    #
    #     return find_best_matches

    def get_params(self):

        nn_params = get_all_params(self.nn_output_layer, trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(self.nn_output_layer)

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(self.nn_output_layer, nn_params_vals)


class GenWaveNetTextWords(object):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.nn_dilations = nn_kwargs['dilations']
        self.nn_dilation_channels = nn_kwargs['dilation_channels']
        self.nn_residual_channels = nn_kwargs['residual_channels']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.nn_input_layers, self.nn_output_layer = self.nn_fn()

    def nn_fn(self):

        l_in_x = InputLayer((None, self.embedding_dim, self.max_length))

        l_in_z = InputLayer((None, self.z_dim))

        l_causal_conv_x = DilatedConv1DLayer(l_in_x, num_filters=self.nn_residual_channels, dilation=1,
                                             nonlinearity=None, b=None)

        l_causal_conv_z = DenseLayer(l_in_z, num_units=self.nn_residual_channels, nonlinearity=None, b=None)
        l_causal_conv_z_rep = RepeatLayer(ReshapeLayer(l_causal_conv_z, ([0], [1], 1,)), self.max_length, axis=-1,
                                          ndim=3)

        l_causal_conv = ElemwiseSumLayer([l_causal_conv_x, l_causal_conv_z_rep])

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

            l_dense = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None,
                                  b=None)

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

    def log_p_z(self, z):
        """
        :param z: (S*N) * dim(z) matrix

        :return log_p_z: (S*N) length vector
        """

        log_p_z = self.dist_z.log_density(z)

        return log_p_z

    def get_probs(self, x, z, all_embeddings, approximate_by_css=False, css_num_samples=None, css_probs=None,
                  mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param all_embeddings: D * E matrix
        :param approximate_by_css: bool
        :param css_samples: css_num_samples length vector
        :param css_is_weights: css_num_samples length vector
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) tensor
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
                probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
            elif mode == 'true':
                probs_numerators -= T.max(probs_denominators, axis=-1)
                probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

                probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
            else:
                raise Exception("mode must be in ['all', 'true']")

        return probs

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

    def generate_text(self, z, all_embeddings, beam=10):
        """
        :param z: N * dim(z) matrix
        :param all_embeddings: D * E matrix

        :return x: N * max(L) tensor
        """

        N = z.shape[0]
        B = beam

        # Initialize the output tensors with (-1) before we fill them in by scan.
        x_init_sampled = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)
        x_init_argmax = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)

        # Step function:
        # l is the number of steps (sentence length)
        # x_prev_sampled is the generated sentence using sample from categorical (from softmax)
        # x_prev_beam is the generated sentence candidates using beam search
        # all_embeddings are the trained embeddings from one-hot to embedding space
        def step(l, x_prev_sampled, x_prev_argmax, z, all_embeddings):
            # Text sampled by using categorical to pick word at each step
            x_prev_sampled_embedded = self.embedder(x_prev_sampled, all_embeddings)  # N * max(L) * E

            probs_sampled = self.get_probs(x_prev_sampled_embedded, z, all_embeddings, mode='all')  # N * max(L) * D

            x_sampled_one_hot = self.dist_x.get_samples([T.shape_padaxis(probs_sampled[:, l], 1)])  # N * 1 * D

            x_sampled_l = T.argmax(x_sampled_one_hot, axis=-1).flatten()  # N

            x_current_sampled = T.set_subtensor(x_prev_sampled[:, l], x_sampled_l)  # N * max(L)

            # Argmax generation
            # TODO: make it from N to B * N
            x_prev_argmax_embedded = self.embedder(x_prev_argmax, all_embeddings)  # N * max(L) * E

            # Get the corresponding probabilities to this x
            probs_argmax = self.get_probs(x_prev_argmax_embedded, z, all_embeddings, mode='all')  # N * max(L) * D

            # Get the character which has the highest probability at step l
            x_argmax_l = T.argmax(probs_argmax[:, l], axis=-1)  # N

            x_current_argmax = T.set_subtensor(x_prev_argmax[:, l], x_argmax_l)  # N * max(L)

            return T.cast(x_current_sampled, 'int32'), T.cast(x_current_argmax, 'int32')


            # Text sampled by using beam search; B is beam size
            # Embedd previously sampled thing

        (x_sampled, x_argmax), updates = theano.scan(step,
                                                     sequences=[T.arange(self.max_length)],
                                                     outputs_info=[x_init_sampled, x_init_argmax],
                                                     non_sequences=[z, all_embeddings],
                                                     )

        # BEAM SEARCH ALGORITHM

        # Initialize the output tensors with (-1) before we fill them in by scan.
        # x_init_beam = T.cast(-1, 'int32') * T.ones((N*B, self.max_length), 'int32')  # N * max(L)

        # Step function:
        # l is the number of steps (sentence length)
        # x_prev_sampled is the generated sentence using sample from categorical (from softmax)
        # x_prev_beam is the generated sentence candidates using beam search
        # all_embeddings are the trained embeddings from one-hot to embedding space
        # def step_beam(l, x_prev_beam, z, all_embeddings):
        #     # Pretend we only have one data point... Add for N different runs afterwards
        #     # Beam search
        #     x_prev_beam_embedded = self.embedder(x_prev_beam, all_embeddings)  # (N*B) * max(L) * E

        #     # Get the corresponding probabilities to this x
        #     # TODO: Fix this
        #     probs_beam = self.get_probs(x_prev_beam_embedded, z, all_embeddings, mode='all')  # (N*B) * max(L) * D

        #     # Get the characters (B of them for each N) which has the highest probability at step l
        #     # TODO: Fix this
        #     x_beam_l = T.argmax(probs_beam[:, l], axis=-1)  # N*B


        #     x_current_beam = T.set_subtensor(x_prev_beam[:, l], x_beam_l)  # (N * B) * max(L)

        #     return T.cast(x_current_beam, 'int32')


        #     # Text sampled by using beam search; B is beam size
        #     # Embedd previously sampled thing

        # Consider N = 1 first
        # fwd are the trace of the probabilities
        # x_init_beam_fwd = T.cast(-1, 'int32') * T.ones((B, self.max_length), 'int32')  # B * max(L)
        # # bwd are the actual words we walk through for each trace
        # x_init_beam_bwd = T.cast(-1, 'int32') * T.ones((B, self.max_length), 'int32')  # B * max(L)

        # def step_beam(l, x_beam_fwd_prev, x_beam_bwd_prev, z, all_embeddings):
        #     # We map up x the the embedding space
        #     x_beam_bwd_prev_embedd = self.embedder(x_beam_bwd_prev, all_embeddings)  # B * max(L) * E

        #     # Get corresponding probabilities for the particular sequences up until this point
        #     # These are log-probabilities due to easier to work with.
        #     probs_beam = T.log(self.get_probs(x_beam_bwd_prev_embedd, z, all_embeddings, mode='all'))  # B * max(L) * D

        #     # Current probabilities over all of the words for each
        #     # TODO: Check if we want to squeeze this to get rid of extra dimension
        #     current_probs = probs_beam[:, l, :]  # B * 1 * D

        #     # Fill in forward step with the probabilities
        #     all_current_scores = 

        #     x_beam_fwd_current = 

        # x_beam, updates = theano.scan(step_beam,
        #                               sequences=[T.arange(self.max_length)],
        #                               outputs_info=[x_init_beam_fwd, x_init_beam_bwd],
        #                               non_sequences=[z, all_embeddings])

        return x_sampled[-1], x_argmax[-1], updates

    def generate_output_prior_fn(self, all_embeddings, num_samples):

        z = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen_sampled, x_gen_argmax],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_output_prior

    def generate_output_posterior_fn(self, x, z, all_embeddings):

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        generate_output_posterior = theano.function(inputs=[x],
                                                    outputs=[z, x_gen_sampled, x_gen_argmax],
                                                    updates=updates,
                                                    allow_input_downcast=True
                                                    )

        return generate_output_posterior

    def get_params(self):

        nn_params = get_all_params(self.nn_output_layer, trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(self.nn_output_layer)

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(self.nn_output_layer, nn_params_vals)


class GenWaveNetTextWordsMultipleZ(GenWaveNetTextWords):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        super(GenWaveNetTextWordsMultipleZ, self).__init__(z_dim, max_length, vocab_size, embedding_dim, embedder,
                                                           dist_z, dist_x, nn_kwargs)

    def nn_fn(self):

        l_in_x = InputLayer((None, self.embedding_dim, self.max_length))

        l_in_z = InputLayer((None, self.z_dim, self.max_length))

        l_causal_conv = DilatedConv1DLayer(l_in_x, num_filters=self.nn_residual_channels, dilation=1, nonlinearity=None)

        l_prev = l_causal_conv

        skip_layers = []

        for h in range(len(self.nn_dilations)):

            l_x_filter = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels,
                                            dilation=self.nn_dilations[h], nonlinearity=None)
            l_z_filter = Conv1DLayer(l_in_z, num_filters=self.nn_dilation_channels, filter_size=1, nonlinearity=None)

            l_filter = NonlinearityLayer(ElemwiseSumLayer([l_x_filter, l_z_filter]), nonlinearity=tanh)

            l_x_gate = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels, dilation=self.nn_dilations[h],
                                          nonlinearity=None)
            l_z_gate = Conv1DLayer(l_in_z, num_filters=self.nn_dilation_channels, filter_size=1, nonlinearity=None)

            l_gate = NonlinearityLayer(ElemwiseSumLayer([l_x_gate, l_z_gate]), nonlinearity=sigmoid)

            l_merge = ElemwiseMergeLayer([l_filter, l_gate], merge_function=T.mul)

            l_dense = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None)

            l_residual = ElemwiseSumLayer([l_prev, l_dense])

            l_skip = Conv1DLayer(l_merge, num_filters=self.embedding_dim, filter_size=1, nonlinearity=None)

            skip_layers.append(l_skip)

            l_prev = l_residual

        l_skip_sum = NonlinearityLayer(ElemwiseSumLayer(skip_layers), nonlinearity=elu)

        l_prev = l_skip_sum

        for h in range(3):

            l_prev = Conv1DLayer(l_prev, num_filters=self.embedding_dim, filter_size=1, nonlinearity=elu)

        l_out = DimshuffleLayer(Conv1DLayer(l_prev, num_filters=self.embedding_dim, filter_size=1, nonlinearity=linear),
                                (0, 2, 1))

        return (l_in_x, l_in_z), l_out

    def get_probs(self, x, z, all_embeddings, approximate_by_css=False, css_num_samples=None, css_probs=None,
                  mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param z: (S*N) * max(L) * dim(z) matrix
        :param all_embeddings: D * E matrix
        :param approximate_by_css: bool
        :param css_samples: css_num_samples length vector
        :param css_is_weights: css_num_samples length vector
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * E tensor
        """

        SN = z.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        hiddens = get_output(self.nn_output_layer, {self.nn_input_layers[0]: x_pre_padded.dimshuffle((0, 2, 1)),
                                                    self.nn_input_layers[1]: z.dimshuffle((0, 2, 1))})
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
                probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
            elif mode == 'true':
                probs_numerators -= T.max(probs_denominators, axis=-1)
                probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

                probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
            else:
                raise Exception("mode must be in ['all', 'true']")

        return probs

    def generate_output_prior_fn(self, all_embeddings, num_samples):

        z = self.dist_z.get_samples(dims=[1, self.max_length, self.z_dim], num_samples=num_samples)  # S * max(L) *
        # dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen_sampled, x_gen_argmax],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_output_prior


class GenWaveNetTextWordsMixtureZ(GenWaveNetTextWords):

    def __init__(self, num_mixtures, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_mu, dist_sigma,
                 dist_z, dist_x, nn_kwargs):

        super(GenWaveNetTextWordsMixtureZ, self).__init__(z_dim, max_length, vocab_size, embedding_dim, embedder,
                                                          dist_z, dist_x, nn_kwargs)

        self.num_mixtures = num_mixtures

        self.dist_mu = dist_mu()
        self.dist_sigma = dist_sigma()

    def log_p_mu(self, mu):
        """
        :param mu: K * dim(z) matrix

        :return log_p_mu: scalar
        """

        log_p_mu = T.sum(self.dist_mu.log_density(mu))  # 1

        return log_p_mu

    def log_p_sigma(self, sigma):
        """
        :param sigma: K * dim(z) matrix

        :return log_p_sigma: scalar
        """

        log_p_sigma = T.sum(self.dist_sigma.log_density(sigma))  # 1

        return log_p_sigma

    def log_p_z(self, z, mu, sigma):
        """
        :param z: (S*N) * dim(z) matrix
        :param mu: K * dim(z) matrix
        :param sigma: K * dim(z) matrix

        :return log_p_z: (S*N) length vector
        """

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.num_mixtures, 1))  # (S*N) * K * dim(z)
        mu_rep = T.tile(T.shape_padleft(mu), (z.shape[0], 1, 1))  # (S*N) * K * dim(z)
        sigma_rep = T.tile(T.shape_padleft(sigma), (z.shape[0], 1, 1))  # (S*N) * K * dim(z)

        # densities = self.dist_z.density(z_rep, [mu_rep, sigma_rep])
        # densities = theano.printing.Print('densities')(densities)
        #
        # log_p_z = T.log((1./self.num_mixtures) * T.sum(T.prod(densities, axis=-1), axis=-1))  # (S*N)

        log_densities = T.sum(self.dist_z.log_density(z_rep, [mu_rep, sigma_rep], sum_trailing_axes=False), axis=-1)
        # (S*N) * K

        log_p_z = T.max(log_densities, axis=-1) + \
                  T.log(T.mean(T.exp(log_densities - T.max(log_densities, axis=-1, keepdims=True)), axis=-1))

        return log_p_z

    def generate_output_prior_fn(self, all_embeddings, num_samples):

        mu = self.dist_mu.get_samples(dims=[self.num_mixtures, self.z_dim], num_samples=1)  # K * dim(z)
        sigma = self.dist_sigma.get_samples(dims=[self.num_mixtures, self.z_dim], num_samples=1)  # K * dim(z)

        mixture_components = random.choice(replace=False,
                                           p=(1./self.num_mixtures) * T.ones((num_samples, self.num_mixtures)))  # S

        means = mu[mixture_components]  # S * dim(z)
        covs = sigma[mixture_components]  # S * dim(z)

        z = self.dist_z.get_samples(num_samples=1, params=[means, covs])  # S * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen_sampled, x_gen_argmax],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_output_prior


class GenWaveNetTextWordsMixtureZLearnHypers(GenWaveNetTextWords):

    def __init__(self, num_mixtures, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        super(GenWaveNetTextWordsMixtureZLearnHypers, self).__init__(z_dim, max_length, vocab_size, embedding_dim,
                                                                     embedder, dist_z, dist_x, nn_kwargs)

        self.num_mixtures = num_mixtures

        self.mixture_means = theano.shared(np.float32(np.random.normal(size=(self.num_mixtures, self.z_dim))))
        self.mixture_covs_exps = theano.shared(np.float32(np.random.normal(size=(self.num_mixtures, self.z_dim))))

        self.mixture_params = [self.mixture_means, self.mixture_covs_exps]

    def log_p_z(self, z):
        """
        :param z: (S*N) * dim(z) matrix

        :return log_p_z: (S*N) length vector
        """

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.num_mixtures, 1))  # (S*N) * K * dim(z)
        mu_rep = T.tile(T.shape_padleft(self.mixture_means), (z.shape[0], 1, 1))  # (S*N) * K * dim(z)
        covs_rep = T.tile(T.shape_padleft(T.exp(self.mixture_covs_exps)), (z.shape[0], 1, 1))  # (S*N) * K * dim(z)

        log_densities = self.dist_z.log_density(z_rep, [mu_rep, covs_rep], sum_trailing_axes=1)  # (S*N) * K

        log_p_z = T.max(log_densities, axis=-1) + \
                  T.log(T.mean(T.exp(log_densities - T.max(log_densities, axis=-1, keepdims=True)), axis=-1))

        return log_p_z

    def generate_output_prior_fn(self, all_embeddings, num_samples):

        mixture_components = random.choice(replace=False,
                                           p=(1./self.num_mixtures) * T.ones((num_samples, self.num_mixtures)))  # S

        means = self.mixture_means[mixture_components]  # S * dim(z)
        covs = T.exp(self.mixture_covs_exps[mixture_components])  # S * dim(z)

        z = self.dist_z.get_samples(num_samples=1, params=[means, covs])  # S * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen_sampled, x_gen_argmax],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_output_prior

    def get_params(self):

        nn_params = get_all_params(self.nn_output_layer, trainable=True)

        return nn_params + self.mixture_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(self.nn_output_layer)

        mixture_params_vals = [p.get_value() for p in self.mixture_params]

        return [nn_params_vals, mixture_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals, mixture_params_vals] = param_values

        set_all_param_values(self.nn_output_layer, nn_params_vals)

        for i in range(len(self.mixture_params)):
            self.mixture_params[i].set_value(mixture_params_vals[i])


class GenWaveNetTextWordsMixtureZLearnHypersMultipleZ(GenWaveNetTextWordsMixtureZLearnHypers):

    def __init__(self, num_mixtures, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        super(GenWaveNetTextWordsMixtureZLearnHypersMultipleZ, self).__init__(num_mixtures, z_dim, max_length,
                                                                              vocab_size, embedding_dim, embedder,
                                                                              dist_z, dist_x, nn_kwargs)

    def nn_fn(self):

        l_in_x = InputLayer((None, self.embedding_dim, self.max_length))

        l_in_z = InputLayer((None, self.z_dim, self.max_length))

        l_causal_conv = DilatedConv1DLayer(l_in_x, num_filters=self.nn_residual_channels, dilation=1, nonlinearity=None)

        l_prev = l_causal_conv

        skip_layers = []

        for h in range(len(self.nn_dilations)):

            l_x_filter = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels,
                                            dilation=self.nn_dilations[h], nonlinearity=None)
            l_z_filter = Conv1DLayer(l_in_z, num_filters=self.nn_dilation_channels, filter_size=1, nonlinearity=None)

            l_filter = NonlinearityLayer(ElemwiseSumLayer([l_x_filter, l_z_filter]), nonlinearity=tanh)

            l_x_gate = DilatedConv1DLayer(l_prev, num_filters=self.nn_dilation_channels, dilation=self.nn_dilations[h],
                                          nonlinearity=None)
            l_z_gate = Conv1DLayer(l_in_z, num_filters=self.nn_dilation_channels, filter_size=1, nonlinearity=None)

            l_gate = NonlinearityLayer(ElemwiseSumLayer([l_x_gate, l_z_gate]), nonlinearity=sigmoid)

            l_merge = ElemwiseMergeLayer([l_filter, l_gate], merge_function=T.mul)

            l_dense = Conv1DLayer(l_merge, num_filters=self.nn_residual_channels, filter_size=1, nonlinearity=None)

            l_residual = ElemwiseSumLayer([l_prev, l_dense])

            l_skip = Conv1DLayer(l_merge, num_filters=self.embedding_dim, filter_size=1, nonlinearity=None)

            skip_layers.append(l_skip)

            l_prev = l_residual

        l_skip_sum = NonlinearityLayer(ElemwiseSumLayer(skip_layers), nonlinearity=elu)

        l_prev = l_skip_sum

        for h in range(3):

            l_prev = Conv1DLayer(l_prev, num_filters=self.embedding_dim, filter_size=1, nonlinearity=elu)

        l_out = DimshuffleLayer(Conv1DLayer(l_prev, num_filters=self.embedding_dim, filter_size=1, nonlinearity=linear),
                                (0, 2, 1))

        return (l_in_x, l_in_z), l_out

    def get_probs(self, x, z, all_embeddings, approximate_by_css=False, css_num_samples=None, css_probs=None,
                  mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param z: (S*N) * max(L) * dim(z) matrix
        :param all_embeddings: D * E matrix
        :param approximate_by_css: bool
        :param css_samples: css_num_samples length vector
        :param css_is_weights: css_num_samples length vector
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * E tensor
        """

        SN = z.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        hiddens = get_output(self.nn_output_layer, {self.nn_input_layers[0]: x_pre_padded.dimshuffle((0, 2, 1)),
                                                    self.nn_input_layers[1]: z.dimshuffle((0, 2, 1))})
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
                probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
            elif mode == 'true':
                probs_numerators -= T.max(probs_denominators, axis=-1)
                probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

                probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
            else:
                raise Exception("mode must be in ['all', 'true']")

        return probs

    def log_p_z(self, z):
        """
        :param z: (S*N) * max(L) * dim(z) matrix

        :return log_p_z: (S*N) length vector
        """

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.num_mixtures, 1, 1))  # (S*N) * K * max(L) * dim(z)
        mu_rep = T.tile(T.shape_padleft(T.shape_padaxis(self.mixture_means, 1)), (z.shape[0], 1, self.max_length, 1))
        # (S*N) * K * max(L) * dim(z)
        covs_rep = T.tile(T.shape_padleft(T.shape_padaxis(T.exp(self.mixture_covs_exps), 1)),
                          (z.shape[0], 1, self.max_length, 1))  # (S*N) * K * max(L) * dim(z)

        log_densities = self.dist_z.log_density(z_rep, [mu_rep, covs_rep], sum_trailing_axes=2)  # (S*N) * K

        log_p_z = T.max(log_densities, axis=-1) + \
                  T.log(T.mean(T.exp(log_densities - T.max(log_densities, axis=-1, keepdims=True)), axis=-1))

        return log_p_z

    def generate_output_prior_fn(self, all_embeddings, num_samples):

        mixture_components = random.choice(replace=False,
                                           p=(1./self.num_mixtures) * T.ones((num_samples*self.max_length,
                                                                              self.num_mixtures))
                                           )  # (S*max(L))

        means = self.mixture_means[mixture_components].reshape((num_samples, self.max_length, self.z_dim))  # S * max(L)
        # * dim(z)
        covs = T.exp(self.mixture_covs_exps[mixture_components]).reshape((num_samples, self.max_length, self.z_dim))
        # S * max(L) * dim(z)

        z = self.dist_z.get_samples(num_samples=1, params=[means, covs])  # S * max(L) * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen_sampled, x_gen_argmax],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_output_prior
