import theano
import theano.tensor as T
import numpy as np
from lasagne.updates import norm_constraint

class SGVBAUTR(object):
    """Class to implement the SGVB from variational autoencoder paper for AUTR generative model"""
    def __init__(self,
                 generative_model,
                 recognition_model,
                 max_len_x,
                 max_len_y,
                 vocab_size_x,
                 vocab_size_y,
                 num_time_steps,
                 gen_nn_kwargs,
                 rec_nn_kwargs,
                 z_dim,
                 z_dist_gen,
                 x_dist_gen,
                 y_dist_gen,
                 z_dist_rec):
        """
        :param generative_model: (model class) generative model (AUTR) instance
        :param recognition_model: (model class) recognition model instance
        :param max_len_x: (int) maximum length of data (x)
        :param max_len_y: (int) maximum length of data (y)
        :param vocab_size_x: (int) vocabulary size of x
        :param vocab_size_y: (int) vocabulary size of y
        :param num_time_steps: (int) number of time steps to run AUTR for
        :param gen_nn_kwargs: (dict) kwargs for generative model
        :param rec_nn_kwargs: (dict) kwargs for recognition model
        :param z_dim: (int) latent dimensionality
        :param z_dist_gen: (distribution class) generative distribution of z
        :param x_dist_gen: (distribution class) generative distribution of x
        :param y_dist_gen: (distribution class) generative distribution of y
        :param z_dist_rec: (distribution class) recognition distribution of z"""
        # set attributes
        self.recognition_model = recognition_model(max_len_x,
                                                   max_len_y,
                                                   vocab_size_x,
                                                   vocab_size_y,
                                                   rec_nn_kwargs,
                                                   z_dim,
                                                   z_dist_rec)

        self.generative_model_x = generative_model(max_len_x,
                                                   vocab_size_x,
                                                   num_time_steps,
                                                   gen_nn_kwargs,
                                                   z_dim,
                                                   z_dist_gen,
                                                   y_dist_gen)

        self.generative_model_y = generative_model(max_len_y,
                                                   vocab_size_y,
                                                   num_time_steps,
                                                   gen_nn_kwargs,
                                                   z_dim,
                                                   z_dist_gen,
                                                   x_dist_gen)
        # max lengths
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y

        # vocab sizes
        self.vocab_size_x = vocab_size_x
        self.vocab_size_y = vocab_size_y

        # dimensions
        self.num_time_steps = num_time_steps
        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs
        self.z_dim = z_dim

        # distributions
        self.z_dist_gen = z_dist_gen
        self.x_dist_gen = x_dist_gen
        self.y_dist_gen = y_dist_gen
        self.z_dist_rec = z_dist_rec

        # One hot encoders. These matrices will be used to map from normal word index form
        # to one hot encoder form. Does it by picking the correct row of the matrix corresponding
        # to the index of the word (Such that we map 0 to a row of zeros)
        self.one_hot_encoder_x = T.concatenate([T.zeros((1, self.vocab_size_x)), T.eye(self.vocab_size_x)], axis=0)
        self.one_hot_encoder_y = T.concatenate([T.zeros((1, self.vocab_size_y)), T.eye(self.vocab_size_y)], axis=0)

    def one_hot_encode(self, x, one_hot_encoder):
        """Transform x from normal to one-hot form

        :param x: ((N * max(L)) tensor) batch matrix of sentences
        :return x_one_hot: ((N * max(L) * D_x)) one-hot form of x"""
        x_one_hot = one_hot_encoder[x]
        return x_one_hot

    def symbolic_elbo(self, x, y, num_samples=1, beta=None, iwae=False):
        """Create the symbolic theano tensors for elbo values

        Create the symbolic tensor for the elbo values, beta is a
        annealing factor.

        :param x: ((N * max(L)) tensor) batch input from x
        :param y: ((N * max(L)) tensor) batch input from y
        :param num_samples: (S int) number of samples to sample
        :param beta: (float) annealing constant for KL term
        :param iwae: (bool) if we are to use Importance Weight Auto Encoder
        :return elbo, kl: (theano functions) returns the elbo and the KL term
        """

        # make the sentences into one hot form
        x_one_hot = self.one_hot_encode(x, self.one_hot_encoder_x)  # N * max(L) * D_x
        y_one_hot = self.one_hot_encode(y, self.one_hot_encoder_y)  # N * max(L) * D_y
        # input the one hot encoded x and y through the recognition model to
        # get the samples of z. S is how many times we sample z in the SGVB
        # approximation of expectation with an average sum.
        z = self.recognition_model.get_samples(x_one_hot, y_one_hot, num_samples)  # (S*N) * dim(z)

        # log-probabilities of q(z|x, y), p(z), p(y|z) given the particular values of z, x_one_hot
        # and y_one_hot
        log_q_z = self.recognition_model.log_q_z(z, x_one_hot, y_one_hot)  # (S*N)
        log_p_z = self.generative_model_x.log_p_z(z)  # (S*N)
        log_p_x = self.generative_model_x.log_p_x(x_one_hot, z)  # (S*N)
        log_p_y = self.generative_model_y.log_p_x(y_one_hot, z)  # (S*N)

        # we calculate the ELBO, the objective function we are trying to optimize
        # IWAE may be used if we specify it (Need to sort out the IWAE part)
        if iwae:
            log_iw_rep = (log_p_x + log_p_y) - (log_p_z - log_q_z)
            log_iw_matrix = log_iw_rep.reshape((num_samples, x.shape[0]))
            # use log-sum-exp trick
            log_iw_max = T.max(log_iw_matrix, axis=0, keepdims=True)
            log_iw_minus_max = log_iw_matrix - log_iw_max

            elbo = T.mean(log_iw_max + T.log(T.mean(T.exp(log_iw_minus_max), axis=0, keepdims=True)))
        else:
            # If beta is not specified we don't use any KL annealing
            # We have that the objective function is the expectation
            # approximated by sampling:
            # L = E_q(z)[log_p_x + log_p_y] - KL[q_z || p_y]
            # This is approximated by:
            # E_q(z)[f(x, y, z)] \approx 1/S \sum_s^S f(x, y, z^(s))
            # where z is sampled from q(z), we also use the reparametrisation trick.
            if beta is None:
                elbo = (1.0 / num_samples) * T.mean((log_p_x + log_p_y) - (log_q_z - log_p_z))
            else:
                elbo = (1.0 / num_samples) * T.mean((log_p_x + log_p_y) - (beta * (log_q_z - log_p_z)))

        # KL divergence term (This should be swapped?)
        kl = (1.0 / num_samples) * T.mean(log_p_z - log_q_z)

        return elbo, kl

    def elbo_fn(self, num_samples, iwae=False):
        """Actual theano elbo function"""
        x = T.imatrix('x')  # N * max(L)
        y = T.imatrix('y')  # N * max(L)
        elbo, kl = self.symbolic_elbo(x, y, num_samples, iwae=iwae)
        elbo_fn = theano.function(inputs=[x, y],
                                  outputs=[elbo, kl],
                                  allow_input_downcast=True)

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None, iwae=False):
        """Optimiser for the AUTR model.

        Calculates the gradients and the updates in order to optimise
        the objective function based on SGVB.

        :param num_samples: (int) the number of samples in the SGVB MC part
        :param grad_norm_constraint: (instance) includes any gradient constraints (such as clipping)
        :param update: (updater) optimiser update function
        :param update_kwargs: (dictionary) kwargs for the update function
        :param saved_update: (bool) if we want to use previously saved updates
        :param iwae: (bool) if we are to use IWAE instead of SGVB
        :return optimiser, updates: returns the optimiser function and the corresponding updates"""
        # input tensors
        x = T.imatrix('x')  # N * max(L)
        y = T.imatrix('y')  # N * max(L)
        # KL annealing tensor
        beta = T.scalar('beta')

        # optimiser tensors
        elbo, kl = self.symbolic_elbo(x, y, num_samples, beta, iwae)
        # all the parameters of the recognition + generative model
        params = self.recognition_model.get_params() + self.generative_model_x.get_params() + self.generative_model_y.get_params()
        # gradients with respect to elbo. Since theano does gradient descent, we minimize the
        # negative objective function instead of optimize the objective function.
        grads = T.grad(-elbo, params)

        # if we have gradient constraints apply them
        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        # add the calculated gradients and parameters to the kwargs of the updater
        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        # get updates from parameters and gradients
        updates = update(**update_kwargs)

        # if we have previously saved updates apply them
        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        # compile the theano function that calculated the elbo while
        # also running optimisation on the parameters of the model.
        optimiser = theano.function(inputs=[x, y, beta],
                                    outputs=[elbo, kl],
                                    updates=updates,
                                    allow_input_downcast=True)

        return optimiser, updates

    def generate_output_prior_fn(self, num_samples, num_samples_per_sample=1, only_final=True):
        """Generate samples from the prior over the latent, p(z)

        :param num_samples: (N int) how many samples to generate
        :param num_samples_per_sample: (S int) how many latents to sample per sample N
        :param only_final: (bool) if we only are to use the final output

        :return prior_output_x: (theano function) prior output function (x)
        :return prior_output_y: (theano function) prior output function (y)"""
        prior_output_x = self.generative_model_x.generate_output_prior_fn(num_samples, num_samples_per_sample, only_final) # N
        prior_output_y = self.generative_model_y.generate_output_prior_fn(num_samples, num_samples_per_sample, only_final) # N

        return prior_output_x, prior_output_y

    def generate_output_posterior_fn(self, num_samples, num_samples_per_sample=1, only_final=True):
        """Generate samples from the posterior over the latent given x, y, q(z|x, y)

        :param num_samples: (N int) how many samples to generate
        :param num_samples_per_sample: (S int) how many latents to sample per sample N
        :param only_final: (bool) if we only are to use the final output

        :return posterior_output_x: (theano function) posterior output function (x)
        :return posterior_output_y: (theano function) posterior output function (y)"""
        # create input matrices
        x = T.imatrix('x')  # N * max(L)
        y = T.imatrix('y')  # N * max(L)
        # one hot encode x and y
        x_one_hot = self.one_hot_encode(x, self.one_hot_encoder_x)
        y_one_hot = self.one_hot_encode(y, self.one_hot_encoder_y)
        # sample z
        z = self.recognition_model.get_samples(x_one_hot, y_one_hot, num_samples, means_only=True)  # N * dim(z) matrix

        # symbolic output functions for x and y
        posterior_output_x = self.generative_model_x.generate_output_posterior_fn(x, y, z, num_samples_per_sample, only_final)
        posterior_output_y = self.generative_model_y.generate_output_posterior_fn(y, x, z, num_samples_per_sample, only_final)

        return posterior_output_x, posterior_output_y

    def follow_latent_trajectory_fn(self, num_samples):
        """Follow latent trajectory in latent space and get output

        :param num_samples: (int) number of samples to generated"""
        alphas = T.vector('alphas')

        # trajectories
        trajectory_x = self.generative_model_x.follow_latent_trajectory_fn(alphas, num_samples)
        trajectory_y = self.generative_model_y.follow_latent_trajectory_fn(alphas, num_samples)

        return trajectory_x, trajectory_y

class SGVBWavenet(object):
    """Class to implement the SGVB from variational autoencoder paper for WaveNetText generative model"""
    def __init__(self,
                 generative_model,
                 recognition_model,
                 max_len_x,
                 max_len_y,
                 vocab_size_x,
                 vocab_size_y,
                 vocab_count_x,
                 vocab_count_y,
                 gen_nn_kwargs,
                 rec_nn_kwargs,
                 z_dim,
                 embedding_dim,
                 z_dist_gen,
                 x_dist_gen,
                 y_dist_gen,
                 z_dist_rec):
        """
        :param generative_model: (model class) generative model (AUTR) instance
        :param recognition_model: (model class) recognition model instance
        :param max_len_x: (int) maximum length of data (x)
        :param max_len_y: (int) maximum length of data (y)
        :param vocab_size_x: (int) vocabulary size of x
        :param vocab_size_y: (int) vocabulary size of y
        :param vocab_count_x: (int array) counts of each word in the x vocabulary
        :param vocab_count_y: (int array) counts of each word in the y vocabulary
        :param num_time_steps: (int) number of time steps to run AUTR for
        :param gen_nn_kwargs: (dict) kwargs for generative model
        :param rec_nn_kwargs: (dict) kwargs for recognition model
        :param z_dim: (int) latent dimensionality
        :param z_dist_gen: (distribution class) generative distribution of z
        :param x_dist_gen: (distribution class) generative distribution of x
        :param y_dist_gen: (distribution class) generative distribution of y
        :param z_dist_rec: (distribution class) recognition distribution of z"""
        # set attributes
        self.recognition_model = recognition_model(max_len_x,
                                                   max_len_y,
                                                   embedding_dim,
                                                   rec_nn_kwargs,
                                                   z_dim,
                                                   z_dist_rec)

        self.generative_model_x = generative_model(max_len_x,
                                                   vocab_size_x,
                                                   gen_nn_kwargs,
                                                   z_dim,
                                                   embedding_dim,
                                                   self.embedder,
                                                   z_dist_gen,
                                                   y_dist_gen)

        self.generative_model_y = generative_model(max_len_y,
                                                   vocab_size_y,
                                                   gen_nn_kwargs,
                                                   z_dim,
                                                   embedding_dim,
                                                   self.embedder,
                                                   z_dist_gen,
                                                   x_dist_gen)

        # max lengths
        self.max_len_x = max_len_x
        self.max_len_y = max_len_y

        # vocab sizes
        self.vocab_size_x = vocab_size_x
        self.vocab_size_y = vocab_size_y

        # vocab counts
        self.vocab_count_x = theano.shared(np.float32(vocab_count_x))
        self.vocab_count_y = theano.shared(np.float32(vocab_count_y))

        # embeddings
        self.all_embeddings_x = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size_x, embedding_dim))))
        self.all_embeddings_y = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size_y, embedding_dim))))

        # dimensions
        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs
        self.z_dim = z_dim
        self.embedding_dim = embedding_dim

        # distributions
        self.z_dist_gen = z_dist_gen
        self.x_dist_gen = x_dist_gen
        self.y_dist_gen = y_dist_gen
        self.z_dist_rec = z_dist_rec

    def embedder(self, x, all_embeddings):
        all_embeddings = T.concatenate([all_embeddings, T.zeros((1, self.embedding_dim))], axis=0)
        return all_embeddings[x]

    def symbolic_elbo(self, x, y, num_samples, beta=None, approximate_by_css=False, css_num_samples=None, translate=False, translation_source=None, iwae=False):
        """Create the symbolic theano tensors for elbo values

        Create the symbolic tensor for the elbo values, beta is a
        annealing factor.

        :param x: ((N * max(L)) tensor) batch input from x
        :param y: ((N * max(L)) tensor) batch input from y
        :param num_samples: (S int) number of samples to sample
        :param beta: (float) annealing constant for KL term
        :param iwae: (bool) if we are to use Importance Weight Auto Encoder
        :return elbo, kl: (theano functions) returns the elbo and the KL term
        """

        # make the sentences into embedding form
        x_embedded = self.embedder(x, self.all_embeddings_x)  # N * max(L) * E_x
        y_embedded = self.embedder(y, self.all_embeddings_y)  # N * max(L) * E_y

        if not translate:
            # input the embedded x and y through the recognition model to
            # get the samples of z. S is how many times we sample z in the SGVB
            # approximation of expectation with an average sum.
            z = self.recognition_model.get_samples(x, y, x_embedded, y_embedded, num_samples)  # (S*N) * dim(z)
            # log-probability of q(z|x, y) given the particular values of z, x_embedded and y_embedded
            log_q_z = self.recognition_model.log_q_z(z, x, y, x_embedded, y_embedded)  # (S*N)
        else:
            z = self.recognition_model.get_samples_translation(x, y, x_embedded, y_embedded, num_samples, translation_source)  # (S*N) * dim(z)
            log_q_z = self.recognition_model.log_q_z_translation(z, x, y, x_embedded, y_embedded, translation_source)  # (S*N)

        # log-probabilities p(x|z) and p(y|z) given the particular values of z, x_embedded and y_embedded
        log_p_z = self.generative_model_x.log_p_z(z)  # (S*N)
        log_p_x = self.generative_model_x.log_p_x(x, z, self.all_embeddings_x, approximate_by_css, css_num_samples, self.vocab_count_x)  # (S*N)
        log_p_y = self.generative_model_y.log_p_x(y, z, self.all_embeddings_y, approximate_by_css, css_num_samples, self.vocab_count_y)  # (S*N)

        # KL divergence
        kl = self.recognition_model.kl_std_gaussian(x, y, x_embedded, y_embedded, translate, translation_source)  # N

        # we calculate the ELBO, the objective function we are trying to optimize
        # IWAE may be used if we specify it (Need to sort out the IWAE part)
        if iwae:
            log_iw_rep = log_p_x + log_p_y + log_p_z - log_q_z
            log_iw_matrix = log_iw_rep.reshape((num_samples, x.shape[0]))
            # use log-sum-exp trick
            log_iw_max = T.max(log_iw_matrix, axis=0, keepdims=True)
            log_iw_minus_max = log_iw_matrix - log_iw_max

            elbo = T.mean(log_iw_max + T.log(T.mean(T.exp(log_iw_minus_max), axis=0, keepdims=True)))
        else:
            # If beta is not specified we don't use any KL annealing
            # We have that the objective function is the expectation
            # approximated by sampling:
            # L = E_q(z)[log_p_x + log_p_y] - KL[q_z || p_y]
            # This is approximated by:
            # E_q(z)[f(x, y, z)] \approx 1/S \sum_s^S f(x, y, z^(s))
            # where z is sampled from q(z), we also use the reparametrisation trick.
            if beta is None:
                elbo = T.sum(((1. / num_samples) * (log_p_x + log_p_y)) - kl)
            else:
                elbo = T.sum(((1. / num_samples) * (log_p_x + log_p_y)) - (beta * kl))

        N = T.shape(log_p_x)

        return elbo, T.sum(kl), T.sum((1. / num_samples) * log_p_x), T.sum((1. / num_samples) * log_p_y)

    def elbo_fn(self, num_samples, approximate_by_css, css_num_samples, translate=False, translation_source=None, iwae=False):
        """Actual theano elbo function"""
        x = T.imatrix('x')  # N * max(L)
        y = T.imatrix('y')  # N * max(L)
        elbo, kl, log_p_x, log_p_y = self.symbolic_elbo(x, y, num_samples, approximate_by_css=approximate_by_css, css_num_samples=css_num_samples,
                                      translate=translate, translation_source=translation_source, iwae=iwae)
        elbo_fn = theano.function(inputs=[x, y],
                                  outputs=[elbo, kl, log_p_x, log_p_y],
                                  allow_input_downcast=True)

        return elbo_fn

    def optimiser(self, num_samples, approximate_by_css, css_num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None, iwae=False):
        """Optimiser for the AUTR model.

        Calculates the gradients and the updates in order to optimise
        the objective function based on SGVB.

        :param num_samples: (int) the number of samples in the SGVB MC part
        :param grad_norm_constraint: (instance) includes any gradient constraints (such as clipping)
        :param update: (updater) optimiser update function
        :param update_kwargs: (dictionary) kwargs for the update function
        :param saved_update: (bool) if we want to use previously saved updates
        :param iwae: (bool) if we are to use IWAE instead of SGVB
        :return optimiser, updates: returns the optimiser function and the corresponding updates"""
        # input tensors
        x = T.imatrix('x')  # N * max(L)
        y = T.imatrix('y')  # N * max(L)
        # KL annealing tensor
        beta = T.scalar('beta')

        # optimiser tensors
        elbo, kl, log_p_x, log_p_y = self.symbolic_elbo(x, y, num_samples, beta, approximate_by_css=approximate_by_css, css_num_samples=css_num_samples, iwae=iwae)
        # all the parameters of the recognition + generative model
        params = self.recognition_model.get_params() + self.generative_model_x.get_params() + self.generative_model_y.get_params()
        # gradients with respect to elbo. Since theano does gradient descent, we minimize the
        # negative objective function instead of optimize the objective function.
        grads = T.grad(-elbo, params)

        # if we have gradient constraints apply them
        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        # add the calculated gradients and parameters to the kwargs of the updater
        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        # get updates from parameters and gradients
        updates = update(**update_kwargs)

        # if we have previously saved updates apply them
        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        # compile the theano function that calculated the elbo while
        # also running optimisation on the parameters of the model.
        optimiser = theano.function(inputs=[x, y, beta],
                                    outputs=[elbo, kl, log_p_x, log_p_y],
                                    updates=updates,
                                    allow_input_downcast=True)

        return optimiser, updates

    def generate_output_prior_fn(self, num_samples, beam_size=5):
        """Generate samples from the prior over the latent, p(z)

        :param num_samples: (N int) how many samples to generate

        :return prior_output_x: (theano function) prior output function (x)
        :return prior_output_y: (theano function) prior output function (y)"""
        prior_output_beam_x = self.generative_model_x.generate_output_prior_fn(self.all_embeddings_x, num_samples, beam_size) # N
        prior_output_beam_y = self.generative_model_y.generate_output_prior_fn(self.all_embeddings_y, num_samples, beam_size) # N

        return prior_output_beam_x, prior_output_beam_y

    def generate_output_posterior_fn(self, num_samples, beam_size=5):
        """Generate samples from the posterior over the latent given x, y, q(z|x, y)

        :param num_samples: (N int) how many samples to generate

        :return posterior_output_beam_x: (theano function) posterior output function (x)
        :return posterior_output_beam_y: (theano function) posterior output function (y)"""
        # create input matrices
        x = T.imatrix('x')  # N * max(L)
        y = T.imatrix('y')  # N * max(L)
        # embedded x and y
        x_embedded = self.embedder(x, self.all_embeddings_x)
        y_embedded = self.embedder(y, self.all_embeddings_y)
        # sample z
        z = self.recognition_model.get_samples(x, y, x_embedded, y_embedded, num_samples, means_only=True)  # N * dim(z) matrix

        # symbolic output functions for x and y
        posterior_output_beam_x = self.generative_model_x.generate_output_posterior_fn(x, y, z, self.all_embeddings_x, beam_size)
        posterior_output_beam_y = self.generative_model_y.generate_output_posterior_fn(y, x, z, self.all_embeddings_y, beam_size)

        return posterior_output_beam_x, posterior_output_beam_y

    def generate_output_translation_fn(self, num_samples, translation_source='x', beam_size=5):
        """Generate samples from the posterior over the latent given x, y, q(z|x, y)

        :param num_samples: (N int) how many samples to generate
        :param translation_source: (int) from which source language to translate

        :return posterior_output_x: (theano function) posterior output function (x)
        :return posterior_output_y: (theano function) posterior output function (y)"""
        # create input matrices
        x = T.imatrix('x')  # N * max(L)
        y = T.imatrix('y')  # N * max(L)
        # embedded x and y
        x_embedded = self.embedder(x, self.all_embeddings_x)
        y_embedded = self.embedder(y, self.all_embeddings_y)
        # sample z
        z = self.recognition_model.get_samples_translation(x, y, x_embedded, y_embedded, num_samples, means_only=True, translation_source=translation_source)  # N * dim(z) matrix

        # symbolic output functions for x and y
        posterior_output_x = self.generative_model_x.generate_output_posterior_fn(x, y, z, self.all_embeddings_x, beam_size)
        posterior_output_y = self.generative_model_y.generate_output_posterior_fn(y, x, z, self.all_embeddings_y, beam_size)

        return posterior_output_x, posterior_output_y
