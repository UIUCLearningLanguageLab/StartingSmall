import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

from starting_small import config
from starting_small.evals import calc_pp, calc_h_term_sims, calc_cluster_score, make_probe_prototype_acts_mat
from starting_small.evals import make_gold


def write_misc_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making misc_summaries...')
    misc_feed_dict = dict()
    misc_feed_dict[graph.train_pp_summary] = calc_pp(hub, graph, sess, False)
    misc_feed_dict[graph.test_pp_summary] = calc_pp(hub, graph, sess, True)
    wx_term_acts = sess.run(graph.wx)
    wx_term_sims = cosine_similarity(wx_term_acts)
    misc_feed_dict[graph.wx_term_sims_summary] = wx_term_sims[np.triu_indices(len(wx_term_sims), k=1)]
    summary = sess.run(graph.misc_summaries, feed_dict=misc_feed_dict)
    summary_writer.add_summary(summary, data_mb)


def write_h_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making h_summaries...')
    h_feed_dict = {}
    for layer_id in range(graph.params.num_layers):
        try:
            h = graph.hs[layer_id]
        except IndexError:
            continue
        h_term_sims = calc_h_term_sims(hub, graph, sess, h)
        name = 'term_sims_layer_{}'.format(layer_id)
        placeholder = graph.h_name2placeholder[name]
        h_feed_dict[placeholder] = h_term_sims[np.triu_indices(len(h_term_sims), k=1)]
    summary = sess.run(graph.h_summaries, feed_dict=h_feed_dict)
    summary_writer.add_summary(summary, data_mb)


def write_cluster_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making cluster_summaries...')
    cluster_feed_dict = dict()
    for layer_id in range(graph.params.num_layers):
        try:
            h = graph.hs[layer_id]
        except IndexError:
            continue
        for hub_mode in config.Eval.hub_modes:
            hub.switch_mode(hub_mode)
            probe_prototype_acts_mat = make_probe_prototype_acts_mat(hub, 'ordered', graph, sess, h)
            probe_sims = cosine_similarity(probe_prototype_acts_mat)
            for cluster_metric in config.Eval.cluster_metrics:
                name = '{}_{}_layer_{}'.format(hub_mode, cluster_metric, layer_id)
                placeholder = graph.cluster_name2placeholder[name]
                cluster_feed_dict[placeholder] = calc_cluster_score(hub, probe_sims, cluster_metric)
    summary = sess.run(graph.cluster_summaries, feed_dict=cluster_feed_dict)
    summary_writer.add_summary(summary, data_mb)


def write_pr_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making pr_summaries...')
    pr_feed_dict = dict()
    for layer_id in range(graph.params.num_layers):
        try:
            h = graph.hs[layer_id]
        except IndexError:
            continue

        for hub_mode in config.Eval.hub_modes:
            hub.switch_mode(hub_mode)
            # sims
            probe_prototype_acts_mat = make_probe_prototype_acts_mat(hub, 'ordered', graph, sess, h)
            probe_sims = cosine_similarity(probe_prototype_acts_mat)
            # labels + predictions
            gold_mat = make_gold(hub)
            pred_mat = np.clip(probe_sims, 0, 1.0, )
            labels = gold_mat[np.triu_indices(len(gold_mat), k=1)]
            predictions = pred_mat[np.triu_indices(len(pred_mat), k=1)]
            # feed placeholders
            name = '{}_pr_layer_{}'.format(hub_mode, layer_id)
            labels_plh, predictions_plh = graph.pr_name2placeholders[name]
            pr_feed_dict[labels_plh] = labels
            pr_feed_dict[predictions_plh] = predictions
            #
            # sess.run(tf.local_variables_initializer())  # why is this needed? it prevents uninitialized variables error
    summary = sess.run(graph.pr_summaries, feed_dict=pr_feed_dict)
    summary_writer.add_summary(summary, data_mb)