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

    def __init__(self, params, hub, wx_mat=None, device=config.Graph.device):
        self.params = params

        # placeholders
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])

        self.cluster_name2placeholder = {}
        for layer_id in range(self.params.num_layers):
            for cluster_metric in config.Eval.cluster_metrics:
                for hub_mode in config.Eval.cluster_metrics:
                    name = '{}_{}_layer_{}'.format(hub_mode, cluster_metric, layer_id)
                    self.cluster_name2placeholder[name] = tf.placeholder(tf.float32)

        self.h_name2placeholder = {}
        for layer_id in range(self.params.num_layers):  # TODO test
            name = 'term_sims_layer_{}'.format(layer_id)
            self.h_name2placeholder[name] = tf.placeholder(tf.float32)

        self.train_pp_summary = tf.placeholder(tf.float32)
        self.test_pp_summary = tf.placeholder(tf.float32)
        self.wx_term_sims_summary = tf.placeholder(tf.float32)

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
            hs = self.rnn_layers(x_embedded)
            final_state = hs[-1]
            # loss
            logit = tf.matmul(final_state, wy) + by
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self.y[:, 0])
            # optimizer
            if self.params.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer
            else:
                optimizer = tf.train.GradientDescentOptimizer

            # public
            self.train_step = optimizer(self.params.lr).minimize(tf.reduce_mean(loss))
            self.softmax_probs = tf.nn.softmax(logit)
            self.pred_ys = tf.argmax(self.softmax_probs, axis=1)
            self.pps = tf.exp(loss)
            self.mean_pp = tf.exp(tf.reduce_mean(loss))  # used too calc test docs pp
            self.h0 = hs[0]
            self.h1 = hs[1] if params.num_layers == 2 else hs[0]
            self.wy = wy
            self.wx = wx
            self.hs = hs
            self.wh = self.cell.trainable_variables
            if self.params.optimizer == 'adagrad':
                self.wh, self.bh, self.wh_adagrad, self.bh_adagrad = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.params.flavor + '0')
                # TODO these vars are for layer 0 only
            else:
                self.wh, self.bh = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.params.flavor + '0')
                self.wh_adagrad, self.bh_adagrad = self.wh, self.bh  # TODO quick fix

        # tensorboard
        with tf.device('cpu'):
            self.misc_summaries = tf.summary.merge([
                tf.summary.histogram('wx_term_sims', self.wx_term_sims_summary),
                tf.summary.scalar('train_pp', self.train_pp_summary),
                tf.summary.scalar('test_pp', self.test_pp_summary),
            ])
            self.h_summaries = tf.summary.merge(
                [tf.summary.scalar(k, v) for k, v in self.h_name2placeholder.items()])
            self.cluster_summaries = tf.summary.merge(
                [tf.summary.scalar(k, v) for k, v in self.cluster_name2placeholder.items()])

            # do this every batch
            self.mean_pp_summary = tf.summary.scalar('mean_pp', self.mean_pp)


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
        hs = []
        for layer_id in range(self.params.num_layers):
            if layer_id > 1:
                raise AttributeError('More than 2 layers are not allowed. Embeddings are retrieved from first two.')
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
            # collect state
            all_states.append(all_state)
            hs.append(h)
        return hs


