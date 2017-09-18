"""
Abstract classes to show how the recognition, generative and other classes
need to be built to work.
"""

import abc

class RecognitionBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,
                 input_max_len,
                 input_vocab_size,
                 nnkwargs,
                 z_dim,
                 z_dist):
    """
    :param input_max_len: (int) maximum length of input data
    :param input_vocab_size: (int) size of the vocabulary from input language
    :param nnkwargs: (dict) dictionary of all of the needed kwargs for the
    neural network used by the recognition model
    :param z_dim: (int) latent dimensionality of z
    :param z_dist: (distribution) latent distribution (will define behaviour of this as well)
    NOTE: z_dime, z_dist have to be the same for recognition model
    and generative model by definition.
    """
    return

    @abc.abstractmethod
    def log_q_z_fn(self, z, x):
        """Log-probabilities of q(z | x) (may assume that q(z | x, y) = q(z | x)q(z | y)
        or just that we concatenate input )

        Calculate the log-probability of q(z | x) elementwise using the distribution of z | x
        which is assumed by the recognition model itself (most of the cases VAE). z | x is already
        sampled from before

        :param z: (N * z_dim tensor) tensor of the samples latents conditioned on x
        :param x: (N * max(L) * D_x tensor) tensor of the batch input

        :return log_q_z_fn: (theano symbolic function) symbolic function of the
        log-probabilities (N vector)
        """
        return

    @abc.abstractmethod
    def get_samples_given_input_fn(self, x, num_samples, means_only):
        """Sample from the latent distribution given input x

        Sample from the latent distribution given passed input tensor x.
        Step before actually getting the log_q_z_fn as we can pass output
        from this function to that function. Also used in sgvb since we need
        to sample z | x a number of times to approximate the expectation by a sum.

        :param x: (N * max(L) * D tensor) input tensor
        :param num_samples: (int) number of samples per input
        :param means_only: (bool) if True, only return means

        :return samples: ((S*N) * z_dim tensor) the samples (might want to change from (S*N)
        to an extra dimension instead)
        """
        return

    @abc.abstractmethod
    def get_params(self):
        """Get the parameters (variables) of the recognition model (i.e. phi)"""
        return

    @abc.abstractmethod
    def get_param_values(self):
        """Get the parameters (values) of the recognition model (i.e. phi)"""
        return

    @abc.abstractmethod
    def set_param_values(self, param_values):
        """Set the parameters of the recognition model (i.e. phi)

        Set the parameters to arbitrary values, for example from saved variables
        from previously trained model.

        :param param_values: (list of np.array) all the parameters needed for the model
        """
        return


class GenerativeBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,
                 output_max_len,
                 output_vocab_size,
                 nnkwargs,
                 z_dim,
                 z_dist,
                 output_dist)
    """
    :param output_max_len: (int) maximum length of output data
    :param output_vocab_size: (int) size of the vocabulary of the output language
    :param nnkwargs: (dict) dictionary of all of the needed kwargs for the
    neural network used by the generative model
    :param z_dim: (int) latent dimensionality of z
    :param z_dist: (distribution) latent distribution (will define behaviour of this as well)
    NOTE: z_dime, z_dist have to be the same for recognition model
    and generative model by definition.
    """

    @abc.abstractmethod
    def log_p_z_fn(self, z):
        """Get the probability p(z) given tensor z, p(z) assumed to be unit normal

        :param z: (N * z_dim tensor) latent tensor from prior distribution"""
        return

    @abc.abstractmethod
    def log_p_output_fn(self, z, y):
        """Get the probability p(y | z) where y is the output.

        :param z: (N * z_dim tensor) latent tensor
        :param y: (N * max(L) * D) output tensor"""
        return

    @abc.abstractmethod
    def output_latent_trajectory_fn(self, num_steps, random_samples):
        """Get the output of the trajectory from z1 to z2

        By following the latent homotopy (linear trajectory
        z = (1 - t)*z1 + t*z2) we get the output. This lets
        us see how the model encode information in the latent space.
        """
        return

    @abc.abstractmethod
    def get_params(self):
        """Get the parameters (variables) of the recognition model (i.e. phi)"""
        return

    @abc.abstractmethod
    def get_param_values(self):
        """Get the parameters (values) of the recognition model (i.e. phi)"""
        return

    @abc.abstractmethod
    def set_param_values(self, param_values):
        """Set the parameters of the recognition model (i.e. phi)

        Set the parameters to arbitrary values, for example from saved variables
        from previously trained model.

        :param param_values: (list of np.array) all the parameters needed for the model
        """
        return


class RunBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,
                 solver,
                 solver_kwargs,
                 recognition_model,
                 generative_model,
                 valid_vocab_x,
                 valid_vocab_y,
                 out_dir,
                 dataset_x,
                 dataset_y,
                 load_param_dir=None,
                 restrict_max_length=None,
                 train_prop=0.95)
        """
        :param solver: solver instance that handles sgvb training and updating
        :param solver_kwargs: kwargs for solver
        :param recognition_model: instance of the recognition model class
        :param generative_model: instance of the generative model class
        :param valid_vocab_x: valid vocabulary for x
        :param valid_vocab_y: valid vocabulary for y
        :param out_dir: path to out directory
        :param dataset_x: path to dataset of x
        :param dataset_y: path to dataset of y
        :param load_param_dir: path to directory of saved variables. If None, train from start
        :param restricted_max_length: restrict the max lengths of the sentences
        :param train_prop: how much of the original data should be split into training/test set
        """
        return

    @abc.abstractmethod
    def train(self):
        """Train the model"""
        return

    @abc.abstractmethod
    def test(self):
        """Test the model"""

    @abc.abstractmethod
    def load_data(self):
        """load data into the run class"""

