import spacy
import tensorflow as tf
import math
import numpy as np
from tensorboard.plugins.pr_curve.summary import op as pr_curve_op

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

        # sim placeholders
        self.sim_name2placeholder = {}
        for hub_mode in config.Eval.hub_modes:
            for word_type in config.Eval.word_types:
                for op_type in config.Eval.op_types:
                    for w_name in config.Eval.w_names:
                        name = '{}_{}_{}_sim_{}'.format(hub_mode, word_type, w_name, op_type)
                        self.sim_name2placeholder[name] = tf.placeholder(tf.float32)

        # ap placeholders
        self.ap_name2placeholder = {}
        for pos in config.Eval.pos_for_map:
            name = 'mean_ap_{}'.format(pos)
            self.ap_name2placeholder[name] = tf.placeholder(tf.float32)

        # cluster placeholders
        self.cluster_name2placeholder = {}
        for layer_id in range(self.params.num_layers):
            for hub_mode in config.Eval.hub_modes:
                for context_type in config.Eval.context_types:
                    for cluster_metric in config.Eval.cluster_metrics:
                        name = '{}_{}_{}_layer_{}'.format(hub_mode, context_type, cluster_metric, layer_id)
                        self.cluster_name2placeholder[name] = tf.placeholder(tf.float32)

        # cluster2 placeholders
        with tf.device('/cpu:0'):  # f1 only works on cpu
            self.cluster2_name2placeholder = {}
            for layer_id in range(self.params.num_layers):
                for hub_mode in config.Eval.hub_modes:
                    name = '{}_tf-f1_labels_layer_{}'.format(hub_mode, layer_id)
                    self.cluster2_name2placeholder[name] = tf.placeholder(tf.bool)
                    name = '{}_tf-f1_predictions_layer_{}'.format(hub_mode, layer_id)
                    self.cluster2_name2placeholder[name] = tf.placeholder(tf.float32)

        # h placeholders
        self.h_name2placeholder = {}
        for layer_id in range(self.params.num_layers):
            for context_type in config.Eval.context_types:
                name = 'h_{}_term_sims_layer_{}'.format(context_type, layer_id)
                self.h_name2placeholder[name] = tf.placeholder(tf.float32)

        # misc placeholder
        self.train_pp_summary = tf.placeholder(tf.float32)
        self.test_pp_summary = tf.placeholder(tf.float32)
        self.wx_term_sims_summary = tf.placeholder(tf.float32)

        # precision + recall placeholders
        self.pr_name2placeholders = {}
        for layer_id in range(self.params.num_layers):
            for hub_mode in config.Eval.hub_modes:
                name = '{}_pr_layer_{}'.format(hub_mode, layer_id)
                self.pr_name2placeholders[name] = [tf.placeholder(t) for t in [tf.bool, tf.float32]]

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
        with tf.device('/cpu:0'):

            # cluster2 computations
            self.cluster2_name2f1_update = {}
            self.cluster2_name2f1 = {}
            self.cluster2_name2initializer = {}
            for layer_id in range(self.params.num_layers):
                for hub_mode in config.Eval.hub_modes:
                    # ops  (f1 calculates f1-score from hidden vars, f1_update updates hidden vars)
                    f1, f1_update = tf.contrib.metrics.f1_score(
                        name='{}_tf-f1-metric_layer_{}'.format(hub_mode, layer_id),
                        labels=self.cluster2_name2placeholder[
                            '{}_tf-f1_labels_layer_{}'.format(hub_mode, layer_id)],
                        predictions=self.cluster2_name2placeholder[
                            '{}_tf-f1_predictions_layer_{}'.format(hub_mode, layer_id)],
                        num_thresholds=config.Eval.num_pr_thresholds)
                    name = '{}_tf-f1_layer_{}'.format(hub_mode, layer_id)
                    self.cluster2_name2f1_update[name] = f1_update
                    self.cluster2_name2f1[name] = f1
                    # save ops + running_vars to dict
                    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                                     scope='{}_tf-f1-metric_layer_{}'.format(hub_mode, layer_id))
                    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
                    self.cluster2_name2initializer[name] = running_vars_initializer

            # summaries
            self.sim_summaries = tf.summary.merge(
                [tf.summary.histogram(k, v) for k, v in self.sim_name2placeholder.items()])
            self.ap_summaries = tf.summary.merge(
                [tf.summary.scalar(k, v) for k, v in self.ap_name2placeholder.items()])
            self.misc_summaries = tf.summary.merge(
                [tf.summary.histogram('wx_term_sims', self.wx_term_sims_summary),
                 tf.summary.scalar('test_pp', self.test_pp_summary)])
            self.h_summaries = tf.summary.merge(
                [tf.summary.histogram(k, v) for k, v in self.h_name2placeholder.items()])
            self.cluster_summaries = tf.summary.merge(
                [tf.summary.scalar(k, v) for k, v in self.cluster_name2placeholder.items()])
            self.cluster2_summaries = tf.summary.merge(
                [tf.summary.scalar(name + '_summary', f1)
                 for name, f1 in self.cluster2_name2f1.items()])
            self.pr_summaries = tf.summary.merge(
                [pr_curve_op(
                    name=name + '_summary',
                    labels=labels,
                    predictions=predictions,
                    num_thresholds=config.Eval.num_pr_thresholds)
                    for name, (labels, predictions) in self.pr_name2placeholders.items()])

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


