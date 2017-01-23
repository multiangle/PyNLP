
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class LTMCell(tf.nn.rnn_cell.LSTMCell):
    def __init__(self,num_units,
                 input_size=None,
                 use_peepholes=True,
                 cell_clip=None,
                 initializer=None,
                 num_proj = None,
                 proj_clip=None,
                 num_unit_shards=1,
                 num_proj_shards=1,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh):
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

        if num_proj:
            self._state_size = (
                tf.nn.rnn_cell.LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            (c_prev,m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with vs.variable_scope(scope or type(self).__name__,
                               initializer=self._initializer):
            concat_w = tf.nn.rnn_cell._get_concat_variable(
                "W", [input_size.value + num_proj, 3 * self._num_units],
                dtype, self._num_unit_shards)

            b = vs.get_variable(
                "B", shape=[3 * self._num_units],
                initializer=init_ops.zeros_initializer, dtype=dtype)

            cell_inputs = array_ops.concat(1,[inputs, m_prev])
            ltm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
            i,j,o = array_ops.split(1,3,ltm_matrix) # i,j,o: [1,num_units]
            c = c_prev + sigmoid(i)*self._activation(j)
            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            m = sigmoid(o) * self._activation(c)
            if self._num_proj is not None:
                concat_w_proj = tf.nn.rnn_cell._get_concat_variable(
                                "W_P", [self._num_units, self._num_proj],
                                dtype, self._num_proj_shards)
                m = math_ops.matmul(m, concat_w_proj)
                if self._proj_clip is not None:
                    m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        new_state = (tf.nn.rnn_cell.LSTMStateTuple(c,m) if self._state_is_tuple
                     else array_ops.concat(1,[c,m]))
        return m, new_state