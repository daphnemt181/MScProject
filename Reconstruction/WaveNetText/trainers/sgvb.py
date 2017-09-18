import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import norm_constraint


class SGVB(object):

    def __init__(self, generative_model, recognition_model, z_dim, max_length, vocab_size, dist_z_gen, dist_x_gen,
                 dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, iwae=False):

        self.vocab_size = vocab_size
        self.max_length = max_length

        self.generative_model = generative_model(z_dim, max_length, vocab_size, dist_z_gen, dist_x_gen, gen_nn_kwargs)
        self.recognition_model = recognition_model(z_dim, max_length, vocab_size, dist_z_rec, rec_nn_kwargs)

        self.iwae = iwae

        self.one_hot_encoder = T.concatenate([T.zeros((1, self.vocab_size)), T.eye(self.vocab_size)], axis=0)

    def one_hot_encode(self, x):

        N = x.shape[0]

        return T.tile(self.one_hot_encoder, (N, 1, 1))[T.arange(N).reshape((N, 1)), x]

    def symbolic_elbo(self, x, num_samples, beta=None):

        x_one_hot = self.one_hot_encode(x)  # N * max(L) * D

        z = self.recognition_model.get_samples(x_one_hot, num_samples)  # (S*N) * dim(z)

        log_q_z = self.recognition_model.log_q_z(z, x_one_hot)  # (S*N)
        log_p_z = self.generative_model.log_p_z(z)  # (S*N)
        log_p_x = self.generative_model.log_p_x(x_one_hot, z)  # (S*N)

        if self.iwae:

            log_iw_rep = log_p_z + log_p_x - log_q_z  # (S*N)

            log_iw_matrix = log_iw_rep.reshape((num_samples, x.shape[0]))  # S * N
            log_iw_max = T.max(log_iw_matrix, axis=0, keepdims=True)  # S * N
            log_iw_minus_max = log_iw_matrix - log_iw_max  # S * N

            elbo = T.sum(log_iw_max + T.log(T.mean(T.exp(log_iw_minus_max), axis=0, keepdims=True)))

        else:

            if beta is None:
                elbo = (1. / num_samples) * T.sum(log_p_z + log_p_x - log_q_z)
            else:
                elbo = (1. / num_samples) * T.sum(log_p_x + (beta * (log_p_z - log_q_z)))

        kl = (1. / num_samples) * T.sum(log_p_z - log_q_z)

        return elbo, kl

    def elbo_fn(self, num_samples):

        x = T.imatrix('x')  # N * max(L)

        elbo, kl = self.symbolic_elbo(x, num_samples)

        elbo_fn = theano.function(inputs=[x],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        x = T.imatrix('x')  # N * max(L)

        beta = T.scalar('beta')

        elbo, kl = self.symbolic_elbo(x, num_samples, beta)

        params = self.generative_model.get_params() + self.recognition_model.get_params()
        grads = T.grad(-elbo, params)

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x, beta],
                                    outputs=[elbo, kl],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_output_prior_fn(self, num_samples):

        return self.generative_model.generate_output_prior_fn(num_samples)

    def generate_output_posterior_fn(self):

        x = T.imatrix('x')  # N * max(L)

        x_one_hot = self.one_hot_encode(x)

        z = self.recognition_model.get_samples(x_one_hot, 1, means_only=True)  # N * dim(z) matrix

        return self.generative_model.generate_output_posterior_fn(x, z)

    # def follow_latent_trajectory_fn(self, num_samples):
    #
    #     alphas = T.vector('alphas')
    #
    #     return self.generative_model.follow_latent_trajectory_fn(alphas, num_samples)
    #
    # def impute_missing_chars(self):
    #
    #     x = T.imatrix('x')  # N * max(L)
    #     missing_chars_mask = T.matrix('missing_chars_mask')  # N * max(L) - 0s indicate character is missing
    #     initial_guess = T.tensor3('initial_guess')  # N * max(L) * D
    #
    #     N = x.shape[0]
    #     D = self.vocab_size
    #
    #     x_one_hot = self.one_hot_encode(x)  # N * max(L) X D
    #
    #     x_one_hot = (T.shape_padright(missing_chars_mask) * x_one_hot) + \
    #                 (T.shape_padright(T.ones_like(missing_chars_mask) - missing_chars_mask) * initial_guess)
    #
    #     z = self.recognition_model.get_samples(x_one_hot, 1, means_only=True)  # N * dim(z) matrix
    #
    #     trans_probs = self.generative_model.get_trans_probs(z)  # N * max(L) * D * D tensor
    #
    #     T1_0 = trans_probs[:, 0, 0]  # N * D matrix
    #     T2_0 = T.zeros((N, D))  # N * D matrix
    #
    #     def step_forward(batch_lm1, batch_l, trans_probs_l, missing_chars_mask_lm1, missing_chars_mask_l, T1_lm1):
    #
    #         T1_l = T.switch(T.shape_padright(T.eq(missing_chars_mask_l, 0)),
    #                         T.max(T.shape_padright(T1_lm1) * trans_probs_l, axis=1),
    #                         T.max(T.shape_padright(T1_lm1) * trans_probs_l, axis=1) * batch_l
    #                         )  # N * D matrix
    #
    #         T2_l = T.switch(T.shape_padright(T.eq(missing_chars_mask_lm1, 0)),
    #                         T.argmax(T.shape_padright(T1_lm1) * trans_probs_l, axis=1),
    #                         T.tile(T.shape_padright(T.argmax(batch_lm1, axis=1)), (1, D))
    #                         )  # N * D matrix
    #
    #         return T.cast(T1_l, 'float32'), T.cast(T2_l, 'float32')
    #
    #     ([T1, T2], _) = theano.scan(step_forward,
    #                                 sequences=[dict(input=x_one_hot.dimshuffle((1, 0, 2)), taps=(-1, 0)),
    #                                            trans_probs[:, 1:].dimshuffle((1, 0, 2, 3)),
    #                                            dict(input=missing_chars_mask.T, taps=(-1, 0))],
    #                                 outputs_info=[T1_0, None],
    #                                 )
    #     # (max(L)-1) * N * D tensors
    #
    #     T1 = T.concatenate([T.shape_padleft(T1_0), T1], axis=0)  # max(L) * N * D tensor
    #     T2 = T.concatenate([T.shape_padleft(T2_0), T2], axis=0)  # max(L) * N * D tensor
    #
    #     char_L = T.cast(T.argmax(T1[-1], axis=1), 'float32')  # N length vector
    #     char_L_one_hot = T.extra_ops.to_one_hot(T.cast(char_L, 'int32'), D)  # N * D matrix
    #
    #     def step_backward(T2_lp1, char_lp1):
    #
    #         char_l = T2_lp1[T.cast(T.arange(N), 'int32'), T.cast(char_lp1, 'int32')]  # N length vector
    #         char_l_one_hot = T.extra_ops.to_one_hot(T.cast(char_l, 'int32'), D)  # N * D matrix
    #
    #         return T.cast(char_l, 'float32'), T.cast(char_l_one_hot, 'float32')
    #
    #     ((chars, chars_one_hot), _) = theano.scan(step_backward,
    #                                               sequences=T2[1:][::-1],
    #                                               outputs_info=[char_L, None],
    #                                               )
    #
    #     chars_one_hot = chars_one_hot[::-1]  # (max(L)-1) * N * D tensor
    #     chars_one_hot = T.concatenate([chars_one_hot, T.shape_padleft(char_L_one_hot)], axis=0)  # max(L) * N * D tensor
    #     chars_one_hot = chars_one_hot.dimshuffle((1, 0, 2))  # N * max(L) * D tensor
    #
    #     impute_missing_chars_fn = theano.function(inputs=[x, missing_chars_mask, initial_guess],
    #                                               outputs=chars_one_hot,
    #                                               allow_input_downcast=True,
    #                                               )
    #
    #     return impute_missing_chars_fn
    #
    # def find_best_matches_fn(self):
    #
    #     sentences = T.imatrix('sentences')  # S * max(L)
    #     batch = T.imatrix('batch')  # N * max(L)
    #
    #     sentences_one_hot = self.one_hot_encode(sentences)  # S * max(L) X D
    #     batch_one_hot = self.one_hot_encode(batch)  # N * max(L) X D
    #
    #     z = self.recognition_model.get_samples(sentences_one_hot, 1, means_only=True)  # S * dim(z) matrix
    #
    #     return self.generative_model.find_best_matches_fn(sentences, sentences_one_hot, batch, batch_one_hot, z)


class SGVBWords(object):

    def __init__(self, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim, vocab_counts,
                 dist_z_gen, dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, iwae=False):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.vocab_counts = theano.shared(np.float32(vocab_counts))
        self.all_embeddings = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))

        self.dist_z_gen = dist_z_gen
        self.dist_x_gen = dist_x_gen
        self.dist_z_rec = dist_z_rec

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_model = self.init_generative_model(generative_model)
        self.recognition_model = self.init_recognition_model(recognition_model)

        self.iwae = iwae

    def init_generative_model(self, generative_model):

        return generative_model(self.z_dim, self.max_length, self.vocab_size, self.embedding_dim, self.embedder,
                                self.dist_z_gen, self.dist_x_gen, self.gen_nn_kwargs)

    def init_recognition_model(self, recognition_model):

        return recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec, self.rec_nn_kwargs)

    def embedder(self, x, all_embeddings):

        all_embeddings = T.concatenate([all_embeddings, T.zeros((1, self.embedding_dim))], axis=0)

        return all_embeddings[x]

    def symbolic_elbo(self, x, num_samples, beta=None, approximate_by_css=False, css_num_samples=None):

        x_embedded = self.embedder(x, self.all_embeddings)  # N * max(L) * E

        z = self.recognition_model.get_samples(x, x_embedded, num_samples)  # (S*N) * dim(z)

        # log_q_z = self.recognition_model.log_q_z(z, x_embedded)  # (S*N)
        # log_p_z = self.generative_model.log_p_z(z)  # (S*N)

        log_p_x = self.generative_model.log_p_x(x, z, self.all_embeddings, approximate_by_css, css_num_samples,
                                                self.vocab_counts)  # (S*N)

        kl = self.recognition_model.kl_std_gaussian(x, x_embedded)  # N

        if self.iwae:

            log_q_z = self.recognition_model.log_q_z(z, x, x_embedded)  # (S*N)
            log_p_z = self.generative_model.log_p_z(z)  # (S*N)

            log_iw_rep = log_p_z + log_p_x - log_q_z  # (S*N)

            log_iw_matrix = log_iw_rep.reshape((num_samples, x.shape[0]))  # S * N
            log_iw_max = T.max(log_iw_matrix, axis=0, keepdims=True)  # S * N
            log_iw_minus_max = log_iw_matrix - log_iw_max  # S * N

            elbo = T.sum(log_iw_max + T.log(T.mean(T.exp(log_iw_minus_max), axis=0, keepdims=True)))

        else:

            if beta is None:
                elbo = T.sum(((1. / num_samples) * log_p_x) - kl)
            else:
                elbo = T.sum(((1. / num_samples) * log_p_x) - (beta * kl))

        return elbo, T.sum(kl)

    def elbo_fn(self, num_samples, approximate_by_css, css_num_samples):

        x = T.imatrix('x')  # N * max(L)

        elbo, kl = self.symbolic_elbo(x, num_samples, approximate_by_css=approximate_by_css,
                                      css_num_samples=css_num_samples)

        elbo_fn = theano.function(inputs=[x],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, approximate_by_css, css_num_samples, grad_norm_constraint, update, update_kwargs,
                  saved_update=None):

        x = T.imatrix('x')  # N * max(L)

        beta = T.scalar('beta')

        elbo, kl = self.symbolic_elbo(x, num_samples, beta, approximate_by_css=approximate_by_css,
                                      css_num_samples=css_num_samples)

        params = self.generative_model.get_params() + self.recognition_model.get_params() + [self.all_embeddings]
        grads = T.grad(-elbo, params)

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x, beta],
                                    outputs=[elbo, kl],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_output_prior_fn(self, num_samples):

        return self.generative_model.generate_output_prior_fn(self.all_embeddings, num_samples)

    def generate_output_posterior_fn(self):

        x = T.imatrix('x')  # N * max(L)

        x_embedded = self.embedder(x, self.all_embeddings)

        z = self.recognition_model.get_samples(x, x_embedded, 1, means_only=True)  # N * dim(z) matrix

        return self.generative_model.generate_output_posterior_fn(x, z, self.all_embeddings)


class SGVBWordsMixtureZ(SGVBWords):

    def __init__(self, generative_model, recognition_model, num_mixtures, z_dim, max_length, vocab_size, embedding_dim,
                 vocab_counts, dist_mu_gen, dist_sigma_gen, dist_z_gen, dist_x_gen, dist_mu_rec, dist_sigma_rec,
                 dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, iwae):

        self.num_mixtures = num_mixtures

        self.dist_mu_gen = dist_mu_gen
        self.dist_sigma_gen = dist_sigma_gen

        self.dist_mu_rec = dist_mu_rec
        self.dist_sigma_rec = dist_sigma_rec

        super(SGVBWordsMixtureZ, self).__init__(generative_model, recognition_model, z_dim, max_length, vocab_size,
                                                embedding_dim, vocab_counts, dist_z_gen, dist_x_gen, dist_z_rec,
                                                gen_nn_kwargs, rec_nn_kwargs, iwae)

    def init_generative_model(self, generative_model):

        return generative_model(self.num_mixtures, self.z_dim, self.max_length, self.vocab_size, self.embedding_dim,
                                self.embedder, self.dist_mu_gen, self.dist_sigma_gen, self.dist_z_gen, self.dist_x_gen,
                                self.gen_nn_kwargs)

    def init_recognition_model(self, recognition_model):

        return recognition_model(self.num_mixtures, self.z_dim, self.max_length, self.embedding_dim, self.dist_mu_rec,
                                 self.dist_sigma_rec, self.dist_z_rec, self.rec_nn_kwargs)

    def symbolic_elbo(self, x, num_samples, beta=None, approximate_by_css=False, css_num_samples=None):

        x_embedded = self.embedder(x, self.all_embeddings)  # N * max(L) * E

        mu, sigma, z = self.recognition_model.get_samples(x_embedded, num_samples)  # (S*N) * dim(z)

        log_q_mu = self.recognition_model.log_q_mu(mu)  # 1
        log_q_sigma = self.recognition_model.log_q_sigma(sigma)  # 1
        log_q_z = self.recognition_model.log_q_z(z, x_embedded)  # (S*N)
        log_p_mu = self.generative_model.log_p_mu(mu)  # 1
        log_p_sigma = self.generative_model.log_p_sigma(sigma)  # 1
        log_p_z = self.generative_model.log_p_z(z, mu, sigma)  # (S*N)
        log_p_x = self.generative_model.log_p_x(x, z, self.all_embeddings, approximate_by_css, css_num_samples,
                                                self.vocab_counts)  # (S*N)

        if beta is None:
            elbo = ((1. / num_samples) * T.sum(log_p_z + log_p_x - log_q_z)) + (log_p_mu + log_p_sigma - log_q_mu -
                                                                                log_q_sigma)
        else:
            elbo = (1. / num_samples) * T.sum(log_p_x + (beta * (log_p_z - log_q_z))) + \
                   (beta * (log_p_mu + log_p_sigma - log_q_mu - log_q_sigma))

        kl = (1. / num_samples) * T.sum(log_p_z - log_q_z)

        return elbo, kl

    def generate_output_posterior_fn(self):

        x = T.imatrix('x')  # N * max(L)

        x_embedded = self.embedder(x, self.all_embeddings)

        mu, sigma, z = self.recognition_model.get_samples(x_embedded, 1, means_only=True)  # N * dim(z) matrix

        return self.generative_model.generate_output_posterior_fn(x, z, self.all_embeddings)


class SGVBWordsMixtureZLearnHypers(SGVBWords):

    def __init__(self, generative_model, recognition_model, num_mixtures, z_dim, max_length, vocab_size, embedding_dim,
                 vocab_counts, dist_z_gen, dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, iwae):

        self.num_mixtures = num_mixtures

        super(SGVBWordsMixtureZLearnHypers, self).__init__(generative_model, recognition_model, z_dim, max_length,
                                                           vocab_size, embedding_dim, vocab_counts, dist_z_gen,
                                                           dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, iwae)

    def init_generative_model(self, generative_model):

        return generative_model(self.num_mixtures, self.z_dim, self.max_length, self.vocab_size, self.embedding_dim,
                                self.embedder, self.dist_z_gen, self.dist_x_gen, self.gen_nn_kwargs)

    def init_recognition_model(self, recognition_model):

        return recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec, self.rec_nn_kwargs)

    def symbolic_elbo(self, x, num_samples, beta=None, approximate_by_css=False, css_num_samples=None):

        x_embedded = self.embedder(x, self.all_embeddings)  # N * max(L) * E

        z = self.recognition_model.get_samples(x_embedded, num_samples)  # (S*N) * dim(z)

        log_q_z = self.recognition_model.log_q_z(z, x_embedded)  # (S*N)
        log_p_z = self.generative_model.log_p_z(z)  # (S*N)

        log_p_x = self.generative_model.log_p_x(x, z, self.all_embeddings, approximate_by_css, css_num_samples,
                                                self.vocab_counts)  # (S*N)

        if self.iwae:

            log_iw_rep = log_p_z + log_p_x - log_q_z  # (S*N)

            log_iw_matrix = log_iw_rep.reshape((num_samples, x.shape[0]))  # S * N
            log_iw_max = T.max(log_iw_matrix, axis=0, keepdims=True)  # S * N
            log_iw_minus_max = log_iw_matrix - log_iw_max  # S * N

            elbo = T.sum(log_iw_max + T.log(T.mean(T.exp(log_iw_minus_max), axis=0, keepdims=True)))

        else:

            if beta is None:
                elbo = (1. / num_samples) * T.sum(log_p_z + log_p_x - log_q_z)
            else:
                elbo = (1. / num_samples) * T.sum(log_p_x + (beta * (log_p_z - log_q_z)))

        kl = (1. / num_samples) * T.sum(log_q_z - log_p_z)

        return elbo, kl
