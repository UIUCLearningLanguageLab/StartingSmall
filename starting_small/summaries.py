import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from starting_small import config
from starting_small.evals import calc_pp
from starting_small.evals import calc_h_term_sims
from starting_small.evals import calc_cluster_score
from starting_small.evals import make_probe_prototype_acts_mat
from starting_small.evals import calc_pos_map
from starting_small.evals import make_gold
from starting_small.evals import calc_diff


def write_sim_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making sim_summaries...')
    fd = dict()
    for hub_mode in config.Eval.hub_modes:
        for word_type in config.Eval.word_types:
            for op_type in config.Eval.op_types:
                for w_name in config.Eval.w_names:
                    name = '{}_{}_{}_sim_{}'.format(hub_mode, word_type, w_name, op_type)
                    placeholder = graph.sim_name2placeholder[name]
                    fd[placeholder] = np.array([calc_diff(hub, graph, sess, cat, word_type, op_type, w_name)
                                                for cat in hub.probe_store.cats])
    summary = sess.run(graph.sim_summaries, feed_dict=fd)
    summary_writer.add_summary(summary, data_mb)


def write_ap_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making ap_summaries...')
    fd = dict()
    for pos in config.Eval.pos_for_map:
        name = 'mean_ap_{}'.format(pos)
        placeholder = graph.ap_name2placeholder[name]
        fd[placeholder] = calc_pos_map(hub, graph, sess, pos)
    summary = sess.run(graph.ap_summaries, feed_dict=fd)
    summary_writer.add_summary(summary, data_mb)


def write_misc_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making misc_summaries...')
    fd = dict()
    fd[graph.test_pp_summary] = calc_pp(hub, graph, sess, True)  # no need for train_pp
    wx_term_acts = sess.run(graph.wx)
    wx_term_sims = cosine_similarity(wx_term_acts)
    fd[graph.wx_term_sims_summary] = wx_term_sims[np.triu_indices(len(wx_term_sims), k=1)]
    summary = sess.run(graph.misc_summaries, feed_dict=fd)
    summary_writer.add_summary(summary, data_mb)


def write_h_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making h_summaries...')
    h_feed_dict = {}
    for layer_id in range(graph.params.num_layers):
        try:
            h = graph.hs[layer_id]
        except IndexError:
            continue
        for context_type in config.Eval.context_types:
            h_term_sims = calc_h_term_sims(hub, context_type, graph, sess, h)
            name = 'h_{}_term_sims_layer_{}'.format(context_type, layer_id)
            placeholder = graph.h_name2placeholder[name]
            h_feed_dict[placeholder] = h_term_sims[np.triu_indices(len(h_term_sims), k=1)]
    summary = sess.run(graph.h_summaries, feed_dict=h_feed_dict)
    summary_writer.add_summary(summary, data_mb)


def write_cluster_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making cluster_summaries...')
    fd = dict()
    for layer_id in range(graph.params.num_layers):
        try:
            h = graph.hs[layer_id]
        except IndexError:
            continue
        for hub_mode in config.Eval.hub_modes:
            hub.switch_mode(hub_mode)
            for context_type in config.Eval.context_types:
                probe_prototype_acts_mat = make_probe_prototype_acts_mat(hub, context_type, graph, sess, h)
                probe_sims = cosine_similarity(probe_prototype_acts_mat)
                for cluster_metric in config.Eval.cluster_metrics:
                    name = '{}_{}_{}_layer_{}'.format(hub_mode, context_type, cluster_metric, layer_id)
                    placeholder = graph.cluster_name2placeholder[name]
                    fd[placeholder] = calc_cluster_score(hub, probe_sims, cluster_metric)
    summary = sess.run(graph.cluster_summaries, feed_dict=fd)
    summary_writer.add_summary(summary, data_mb)


def write_cluster2_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making cluster2_summaries...')
    fd = dict()
    for layer_id in range(graph.params.num_layers):
        try:
            h = graph.hs[layer_id]
        except IndexError:
            continue
        for hub_mode in config.Eval.hub_modes:
            hub.switch_mode(hub_mode)
            # get placeholders
            name = '{}_tf-f1_labels_layer_{}'.format(hub_mode, layer_id)
            labels_placeholder = graph.cluster2_name2placeholder[name]
            name = '{}_tf-f1_predictions_layer_{}'.format(hub_mode, layer_id)
            predictions_placeholder = graph.cluster2_name2placeholder[name]
            # labels
            gold_mat = make_gold(hub)
            labels = gold_mat[np.triu_indices(len(gold_mat), k=1)]
            # predictions
            probe_prototype_acts_mat = make_probe_prototype_acts_mat(hub, 'ordered', graph, sess, h)
            probe_sims = cosine_similarity(probe_prototype_acts_mat)
            probe_sims_clipped = np.clip(probe_sims, 0, 1)
            predictions = probe_sims_clipped[np.triu_indices(len(probe_sims_clipped), k=1)]
            # feed
            fd[labels_placeholder] = labels
            fd[predictions_placeholder] = predictions
            # update hidden/running variables
            name = '{}_tf-f1_layer_{}'.format(hub_mode, layer_id)
            init = graph.cluster2_name2initializer[name]
            sess.run(init)  # reset the hidden variables associated with f1
            f1_update = graph.cluster2_name2f1_update[name]
            sess.run(f1_update, feed_dict=fd)
            # calculate new value
            f1 = graph.cluster2_name2f1[name]
            sess.run(f1)
    summary = sess.run(graph.cluster2_summaries, feed_dict=fd)
    summary_writer.add_summary(summary, data_mb)


def write_pr_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making pr_summaries...')
    fd = dict()
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
            fd[labels_plh] = labels
            fd[predictions_plh] = predictions
    summary = sess.run(graph.pr_summaries, feed_dict=fd)
    summary_writer.add_summary(summary, data_mb)

