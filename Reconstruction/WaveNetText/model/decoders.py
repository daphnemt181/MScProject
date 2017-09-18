import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


random = RandomStreams()


def get_probs(chars, trans_probs):
    """
    :param chars: N * max(L) matrix
    :param trans_probs: N * max(L) * D * D tensor
    """

    N = trans_probs.shape[0]

    def step(char_lm1, char_l, trans_probs_l):

        char_lm1 = T.cast(char_lm1, 'int32')
        char_l = T.cast(char_l, 'int32')

        return trans_probs_l[T.arange(N), char_lm1, char_l]  # N

    probs, _ = theano.scan(step,
                           sequences=[dict(input=T.concatenate([T.zeros((N, 1)), chars], axis=1).T, taps=[-1, -0]),
                                      trans_probs.dimshuffle((1, 0, 2, 3))],
                           )
    # max(L) * N matrix

    return probs.T


def sample(trans_probs, num_samples):
    """
    :param trans_probs: N * max(L) * D * D
    :param num_samples: int
    """

    N = trans_probs.shape[0]

    trans_probs = trans_probs.repeat(num_samples, axis=0)  # (S*N) * max(L) * D * D
    chars_init = T.zeros((num_samples * N,))  # (S*N)

    def step(trans_probs_l, chars_lm1):

        probs_l = trans_probs_l[T.arange(trans_probs_l.shape[0]), T.cast(chars_lm1, 'int32')]  # (S*N) * D

        chars_l_one_hot = random.multinomial(pvals=probs_l)  # (S*N) * D

        chars_l = T.argmax(chars_l_one_hot, axis=1)  # (S*N)

        return T.cast(chars_l, 'float32')

    chars, updates = theano.scan(step,
                                 sequences=[trans_probs.dimshuffle((1, 0, 2, 3))],
                                 outputs_info=[chars_init],
                                 )
    # max(L) * (S*N)

    chars = chars.T  # (S*N) * max(L)

    probs = get_probs(chars, trans_probs)  # (S*N) * max(L)

    return chars, probs, updates  # (S*N) * max(L) and (S*N) * max(L) and updates


def viterbi(trans_probs):
    """
    :param trans_probs: N * max(L) * D * D tensor
    """

    N = trans_probs.shape[0]
    D = trans_probs.shape[-1]

    T1_0 = trans_probs[:, 0, 0]  # N * D matrix

    T2_0 = T.zeros((N, D))  # N * D matrix

    def step_forward(trans_probs_l, T1_lm1):

        T1_l = T.max(T.shape_padright(T1_lm1) * trans_probs_l, axis=1)  # N * D matrix

        T2_l = T.argmax(T.shape_padright(T1_lm1) * trans_probs_l, axis=1)  # N * D matrix

        return T.cast(T1_l, 'float32'), T.cast(T2_l, 'float32')

    ([T1, T2], _) = theano.scan(step_forward,
                                sequences=trans_probs[:, 1:].dimshuffle((1, 0, 2, 3)),
                                outputs_info=[T1_0, None],
                                )
    # (max(L)-1) * N * D tensors

    T1 = T.concatenate([T.shape_padleft(T1_0), T1], axis=0)  # max(L) * N * D
    T2 = T.concatenate([T.shape_padleft(T2_0), T2], axis=0)  # max(L) * N * D

    char_L = T.cast(T.argmax(T1[-1], axis=1), 'float32')  # N

    def step_backward(T2_lp1, char_lp1):

        char_l = T2_lp1[T.arange(N), T.cast(char_lp1, 'int32')]  # N

        return T.cast(char_l, 'float32')

    chars, _ = theano.scan(step_backward,
                           sequences=T2[1:][::-1],
                           outputs_info=[char_L],
                           )
    # (max(L)-1) * N

    chars = chars[::-1]  # (max(L)-1) * N

    chars = T.concatenate([chars, T.shape_padleft(char_L)], axis=0).T  # N * max(L)

    probs = get_probs(chars, trans_probs)  # N * max(L)

    return chars, probs  # N * max(L) and N * max(L)


def beam_search(trans_probs):
    """
    :param trans_probs: N * max(L) * D * D tensor
    """

    N = trans_probs.shape[0]
    D = trans_probs.shape[-1]

    T1_0 = trans_probs[:, 0, 0]  # N * D matrix

    T2_0 = T.zeros((N, D))  # N * D matrix

    def step_forward(trans_probs_l, T1_lm1):

        T1_l = T.max(T.shape_padright(T1_lm1) * trans_probs_l, axis=1)  # N * D matrix

        T2_l = T.argmax(T.shape_padright(T1_lm1) * trans_probs_l, axis=1)  # N * D matrix

        return T.cast(T1_l, 'float32'), T.cast(T2_l, 'float32')

    ([T1, T2], _) = theano.scan(step_forward,
                                sequences=trans_probs[:, 1:].dimshuffle((1, 0, 2, 3)),
                                outputs_info=[T1_0, None],
                                )
    # (max(L)-1) * N * D tensors

    T1 = T.concatenate([T.shape_padleft(T1_0), T1], axis=0)  # max(L) * N * D
    T2 = T.concatenate([T.shape_padleft(T2_0), T2], axis=0)  # max(L) * N * D

    char_L = T.cast(T.argmax(T1[-1], axis=1), 'float32')  # N

    def step_backward(T2_lp1, char_lp1):

        char_l = T2_lp1[T.arange(N), T.cast(char_lp1, 'int32')]  # N

        return T.cast(char_l, 'float32')

    chars, _ = theano.scan(step_backward,
                           sequences=T2[1:][::-1],
                           outputs_info=[char_L],
                           )
    # (max(L)-1) * N

    chars = chars[::-1]  # (max(L)-1) * N

    chars = T.concatenate([chars, T.shape_padleft(char_L)], axis=0).T  # N * max(L)

    probs = get_probs(chars, trans_probs)  # N * max(L)

    return chars, probs  # N * max(L) and N * max(L)
