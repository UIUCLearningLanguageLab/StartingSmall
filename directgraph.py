import spacy
import tensorflow as tf
import math
import numpy as np

from starting_small import config
from starting_small.graphutils import DirectLSTMCell
from starting_small.graphutils import DirectMultLSTMCell
from starting_small.graphutils import DirectRNNCell
from starting_small.graphutils import DirectDeltaRNNCell
from starting_small.graphutils import DirectFahlmanRNNCell


class DirectGraph:
    """
    Defines a tensorflow graph of a recurrent neural network
    """

    def __init__(self,
                 params,
                 hub,
                 wx_mat=None,
                 device=config.Graph.device):
        self.params = params

        # placeholders
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.y_attention = tf.placeholder(tf.float32, shape=[None, None])

        # weights
        with tf.device('/cpu:0'):  # embedding op works on cpu only
            if wx_mat is not None:
                print('Initializing embeddings with wx_mat')
                wx = tf.get_variable('Wx',
                                     [wx_mat.shape[0], wx_mat.shape[1]],
                                     initializer=tf.constant_initializer(wx_mat))
            elif self.params.wx_init == 'glove300':
                print('Initializing embeddings with "en_core_web_lg"')
                nlp = spacy.load('en_core_web_lg')
                glove_embed_size = 300
                embeddings = np.zeros((hub.train_terms.num_types, self.params.embed_size))
                for term, term_id in hub.train_terms.term_id_dict.items():
                    embeddings[term_id, :glove_embed_size] = nlp.vocab[term].vector
                wx = tf.get_variable('Wx',
                                     [embeddings.shape[0], embeddings.shape[1]],
                                     trainable=False,
                                     initializer=tf.constant_initializer(embeddings))
            elif self.params.wx_init == 'random':
                print('Initializing embeddings with "truncated normal initializer"')
                wx = tf.get_variable('Wx',
                                     [self.params.num_types, self.params.embed_size],
                                     initializer=tf.truncated_normal_initializer(
                                         stddev=1.0 / math.sqrt(self.params.embed_size * self.multi)))
            else:
                raise AttributeError('starting_small: Invalid arg to "wx_init".')
        with tf.device(device):
            last_cell_size = self.params.embed_size
            wy = tf.get_variable('Wy',
                                 [last_cell_size, self.params.num_types],
                                 initializer=tf.truncated_normal_initializer(
                                     stddev=1.0 / math.sqrt(self.params.embed_size * self.multi)))
            by = tf.get_variable('by',
                                 [self.params.num_types],
                                 initializer=tf.zeros_initializer())

        # ops
        with tf.device(device):
            # rnn
            x_embedded = tf.nn.embedding_lookup(wx, self.x)
            final_state, representation = self.rnn_layers(x_embedded)
            # loss
            logit = tf.matmul(final_state, wy) + by
            attention_3d = tf.tile(tf.expand_dims(self.y_attention, -1), [1, 1, self.params.num_types])
            eyes = tf.eye(self.params.num_types, self.params.num_types, dtype=tf.float32)
            y_eyed_3d = tf.nn.embedding_lookup(eyes, self.y)  # [mb, num_y, num_inputs]
            y_eyed_3d_attended = y_eyed_3d * attention_3d
            y_eyed_att = tf.reduce_sum(y_eyed_3d_attended, axis=1)  # [mb, num_inputs]
            att_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_eyed_att)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self.y[:, 0])
            # optimizer
            if self.params.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer
            else:
                optimizer = tf.train.GradientDescentOptimizer
            # public
            self.train_step = optimizer(self.params.lr).minimize(tf.reduce_mean(att_loss))
            self.softmax_probs = tf.nn.softmax(logit)
            self.pred_ys = tf.argmax(self.softmax_probs, axis=1)
            self.pps = tf.exp(loss)
            self.mean_pp = tf.exp(tf.reduce_mean(loss))  # used too calc test docs pp
            self.representation = representation
            self.wy = wy
            self.wx = wx
            self.wh = self.cell.trainable_variables
            if self.params.optimizer == 'adagrad':
                self.wh, self.bh, self.wh_adagrad, self.bh_adagrad = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.params.flavor + '0')
                # TODO these vars are for layer 0 only
            else:
                self.wh, self.bh = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.params.flavor + '0')
                self.wh_adagrad, self.bh_adagrad = self.wh, self.bh  # TODO quick fix


    @property
    def cell(self):
        if self.params.flavor == 'rnn':
            cell = DirectRNNCell
        elif self.params.flavor == 'fahlmanrnn':
            cell = DirectFahlmanRNNCell
        elif self.params.flavor == 'lstm':
            cell = DirectLSTMCell
        elif self.params.flavor == 'mulstm':
            cell = DirectMultLSTMCell
        elif self.params.flavor == 'delta':
            cell = DirectDeltaRNNCell
        else:
            raise AttributeError('starting_small: Invalid arg to "flavor".')
        return cell

    @property
    def multi(self):
        if self.params.flavor in ['lstm', 'mulstm']:
            multi = 4  # 4 gates
        elif self.params.flavor == 'deltarnn':
            multi = 2
        else:
            multi = 1
        return multi

    def rnn_layers(self, x_embedded):
        all_states = []
        final_states = []
        for layer_id in range(self.params.num_layers):
            print('Making layer {}...'.format(layer_id))
            # prev_layer
            try:
                prev_layer = all_states[-1]
            except IndexError:
                prev_layer = x_embedded
            # cell
            cell_input_ = prev_layer
            cell_input = tf.tile(cell_input_, [1, 1, self.multi])  # directlstm requires multiple copies of input
            cell_size = self.params.embed_size
            cell_ = self.cell(cell_size)
            # calc state
            all_state, (c, h) = tf.nn.dynamic_rnn(
                cell_, cell_input, dtype=tf.float32, scope=self.params.flavor + str(layer_id))  # TODO test scope
            final_state = h  # this no longer has flexibility to use tf.tanh(c)
            # collect state
            all_states.append(all_state)
            final_states.append(final_state)
        # result
        representation = final_states[self.params.rep_layer_id]
        final_state = final_states[-1]
        return final_state, representation


