
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple


class DirectRNNCell(RNNCell):
    """
    RNN cell without input weights.
    """

    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    def __call__(self, input, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # computation
            c_prev, h_prev = state
            with tf.variable_scope("new_h"):
                rec_input = tf.contrib.layers.fully_connected(inputs=h_prev,
                                                              num_outputs=self._num_units,
                                                              biases_initializer=tf.constant_initializer(0.0),
                                                              activation_fn=None)
            new_h = tf.nn.tanh(rec_input + input)
        # new_c, new_h
        new_c = new_h
        new_h = new_h
        new_state = (LSTMStateTuple(new_c, new_h))
        return new_h, new_state


@tf.RegisterGradient('TanhPrimeFahlman')
def tanh_prime_fahlman(unused_op, grad):
    return tf.subtract(tf.constant(1.0), tf.square(tf.tanh(grad))) + 0.1


class DirectFahlmanRNNCell(RNNCell):  # TODO does not work
    """
    RNN cell without input weights.
    """

    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    def __call__(self, input, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # computation
            c_prev, h_prev = state
            with tf.variable_scope("new_h"):
                rec_input = tf.contrib.layers.fully_connected(inputs=h_prev,
                                                              num_outputs=self._num_units,
                                                              biases_initializer=tf.constant_initializer(0.0),
                                                              activation_fn=None)
            with tf.get_default_graph().gradient_override_map({'Tanh': 'TanhPrimeFahlman'}):
                new_h = tf.nn.tanh(rec_input + input)
        # new_c, new_h
        new_c = new_h
        new_h = new_h
        new_state = (LSTMStateTuple(new_c, new_h))
        return new_h, new_state


class DirectMultLSTMCell(RNNCell):  # TOD0 test
    """
    Multiplicative LSTM
    """

    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units
        self._forget_bias = 1.0

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    def __call__(self, input, state, scope=None):  # TODO test
        with tf.variable_scope(scope or type(self).__name__):
            # computation
            c_prev, h_prev = state
            with tf.variable_scope('mul'):
                concat = tf.contrib.layers.fully_connected(inputs=[input, h_prev],
                                                           num_outputs=2 * self._num_units,
                                                           biases_initializer=tf.constant_initializer(0.0),
                                                           activation_fn=None)
            proj_input, rec_input = tf.split(value=concat, num_or_size_splits=2, axis=1)
            mul_input = proj_input * rec_input  # equation (18)
            with tf.variable_scope('rec_input'):
                rec_mul_input = tf.contrib.layers.fully_connected(inputs=mul_input,
                                                                  num_outputs=4 * self._num_units,
                                                                  biases_initializer=tf.constant_initializer(0.0),
                                                                  activation_fn=None)
                b = tf.get_variable('b', [self._num_units * 4])
            lstm_mat = input + rec_mul_input + b
            i, j, f, o = tf.split(value=lstm_mat, num_or_size_splits=4, axis=1)
        # new_c, new_h
        new_c = (c_prev * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * tf.nn.tanh(j))
        new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)
        new_state = (LSTMStateTuple(new_c, new_h))
        return new_h, new_state


class DirectLSTMCell(RNNCell):
    """
    LSTM without input weights.
    """
    def __init__(self, num_units, forget_bias=1.0, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units
        self.forget_bias = forget_bias

    @property
    def input_size(self):
        return self._num_units

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, input, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # computation
            c_prev, h_prev = state
            with tf.variable_scope('rec_input'):
                rec_input = tf.contrib.layers.fully_connected(inputs=h_prev,
                                                              num_outputs=4 * self._num_units,
                                                              biases_initializer=tf.constant_initializer(0.0),
                                                              activation_fn=None)
                b = tf.get_variable('b', [self._num_units * 4])
            lstm_mat = input + rec_input + b
            i, j, f, o = tf.split(value=lstm_mat, num_or_size_splits=4, axis=1)
        # new_c, new_h
        new_c = c_prev * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * tf.nn.tanh(j)
        new_h = tf.sigmoid(o) * tf.nn.tanh(new_c)
        new_state = (LSTMStateTuple(new_c, new_h))
        return new_h, new_state


class DirectDeltaRNNCell(RNNCell):
    """
    Delta RNN Cell without input weights
    """
    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    def _outer_function(self, inner_function_output, inputs2, state, wx_parameterization_gate=True):
        """Simulates Equation 3 in Delta RNN paper
        r, the gate, can be parameterized in many different ways.
        """
        assert inner_function_output.get_shape().as_list() == \
               state.get_shape().as_list()
        with tf.variable_scope("OuterFunction"):
            r_bias = tf.get_variable(
                "outer_function_gate",
                [self._num_units],
                dtype=tf.float32, initializer=tf.zeros_initializer())

            # Equation 5 in Delta Rnn Paper
            if wx_parameterization_gate:
                r = inputs2 + r_bias
            else:
                r = r_bias
            gate = tf.nn.sigmoid(r)
            output = tf.nn.tanh((1.0 - gate) * inner_function_output + gate * state)
        return output

    def _inner_function(self, inputs1, state):
        """second order function as described equation 11 in delta rnn paper
        The main goal is to produce z_t of this function
        """
        with tf.variable_scope("InnerFunction"):
            # recurrence
            with tf.variable_scope("new_h"):
                new_h = tf.contrib.layers.fully_connected(inputs=state,
                                                          num_outputs=self._num_units,
                                                          biases_initializer=tf.constant_initializer(0.0),
                                                          activation_fn=None)
            alpha = tf.get_variable(
                "alpha", [self._num_units], dtype=tf.float32,
                initializer=tf.constant_initializer(1.0))
            beta_one = tf.get_variable(
                "beta_one", [self._num_units], dtype=tf.float32,
                initializer=tf.constant_initializer(1.0))
            beta_two = tf.get_variable(
                "beta_two", [self._num_units], dtype=tf.float32,
                initializer=tf.constant_initializer(1.0))
            z_t_bias = tf.get_variable(
                "z_t_bias", [self._num_units], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
            # Second Order Cell Calculations
            d_1_t = alpha * new_h * inputs1
            d_2_t = beta_one * new_h + beta_two * inputs1
            z_t = tf.nn.tanh(d_1_t + d_2_t + z_t_bias)
        return z_t

    def __call__(self, input, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            inputs1, inputs2 = tf.split(value=input, num_or_size_splits=2, axis=1)
            inner_function_output = self._inner_function(inputs1, state)
            new_h = self._outer_function(inner_function_output, inputs2, state)
        # new_c, new_h
        new_c = new_h
        new_h = new_h
        new_state = (LSTMStateTuple(new_c, new_h))
        return new_h, new_state

