import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from starting_small import config
from starting_small.evals import calc_pp, calc_h_term_sims, calc_cluster_score, make_probe_prototype_acts_mat


def write_misc_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making misc_summaries...')
    misc_feed_dict = dict()
    misc_feed_dict[graph.train_pp_summary] = calc_pp(hub, graph, sess, False) if not config.Eval.skip_pp else np.nan
    misc_feed_dict[graph.test_pp_summary] = calc_pp(hub, graph, sess, True) if not config.Eval.skip_pp else np.nan
    wx_term_acts = sess.run(graph.wx)
    wx_term_sims = cosine_similarity(wx_term_acts)
    wx_term_sims_triu = wx_term_sims[np.triu_indices(len(wx_term_sims), k=1)]
    misc_feed_dict[graph.wx_term_sims_summary] = wx_term_sims_triu
    summary = sess.run(graph.misc_summaries, feed_dict=misc_feed_dict)
    summary_writer.add_summary(summary, data_mb)


def write_h_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making h_summaries...')
    h_feed_dict = {}
    for layer_id, sims_plhs in zip([0, 1], [graph.h0_term_sims_summary, graph.h1_term_sims_summary]):
        try:
            h = graph.hs[layer_id]
        except IndexError:
            continue
        h_term_sims = calc_h_term_sims(hub, graph, sess, h) if not config.Eval.skip_term_sims else \
            np.random.uniform(-1.0, 1.0, (hub.params.num_types, hub.params.num_types))
        name = 'term_sims_layer_{}'.format(layer_id)
        placeholder = graph.h_name2placeolder[name]
        h_feed_dict[placeholder] = h_term_sims[np.triu_indices(len(h_term_sims), k=1)]
    summary = sess.run(graph.h_summaries, feed_dict=h_feed_dict)
    summary_writer.add_summary(summary, data_mb)


def write_cluster_summaries(hub, graph, sess, data_mb, summary_writer):
    print('Making cluster_summaries...')
    cluster_feed_dict = dict()
    cluster_placeholders0 = [(graph.sem_ba_summary0, graph.sem_f1_summary0, graph.sem_ck_summary0),
                             (graph.syn_ba_summary0, graph.syn_f1_summary0, graph.syn_ck_summary0)]
    cluster_placeholders1 = [(graph.sem_ba_summary1, graph.sem_f1_summary1, graph.sem_ck_summary1),
                             (graph.syn_ba_summary1, graph.syn_f1_summary1, graph.syn_ck_summary1)]
    for layer_id, cluster_plhs in zip( [0, 1], [cluster_placeholders0, cluster_placeholders1]):
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
                placeholder = graph.name2placeholder[name]
                cluster_feed_dict[placeholder] = calc_cluster_score(hub, probe_sims, cluster_metric)
    summary = sess.run(graph.cluster_summaries, feed_dict=cluster_feed_dict)
    summary_writer.add_summary(summary, data_mb)
