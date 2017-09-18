import theano.tensor as T
from lasagne.theano_extensions import conv
from lasagne import init, nonlinearities
from lasagne.layers import Conv1DLayer, Layer, PadLayer


class DilatedConv1DLayer(Conv1DLayer):

    def __init__(self, incoming, num_filters, dilation=1, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify, flip_filters=False,
                 convolution=conv.conv1d_mc0, **kwargs):

        self.dilation = dilation

        filter_size = 2
        filter_size += dilation - 1

        l_pad = PadLayer(incoming, batch_ndim=2, width=[(dilation, 0)])

        super(DilatedConv1DLayer, self).__init__(incoming=l_pad, num_filters=num_filters, filter_size=filter_size,
                                                 stride=1, pad=0, untie_biases=untie_biases, W=W, b=b,
                                                 nonlinearity=nonlinearity, flip_filters=flip_filters,
                                                 convolution=convolution, **kwargs)

    def convolve(self, input, **kwargs):

        border_mode = 'half' if self.pad == 'same' else self.pad

        if self.dilation > 1:
            mask = T.zeros(self.get_W_shape())
            mask = T.set_subtensor(mask[:, :, 0], 1)
            mask = T.set_subtensor(mask[:, :, -1], 1)

        else:
            mask = T.ones_like(self.W)

        conved = self.convolution(input, self.W * mask,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)

        return conved


class RepeatLayer(Layer):

    def __init__(self, incoming, repeats, axis, ndim, name=None):

        self.repeats = repeats
        self.axis = axis
        self.ndim = ndim

        super(RepeatLayer, self).__init__(incoming=incoming, name=name)

    def get_output_shape_for(self, input_shape):

        output_shape = list(input_shape)
        output_shape[self.axis] *= self.repeats

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):

        repeat_pattern = [1] * self.ndim
        repeat_pattern[self.axis] = self.repeats

        return T.tile(input, repeat_pattern)

