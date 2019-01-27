import pyprind
import numpy as np
from bayes_opt import BayesianOptimization
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity

from starting_small import config


def check_nans(mat, name='mat'):
    if np.any(np.isnan(mat)):
        num_nans = np.sum(np.isnan(mat))
        print('Found {} Nans in {}'.format(num_nans, name), 'red')


def make_probe_prototype_acts_mat(hub, context_type, graph, sess, h):
    print('Making "{}" "{}" probe prototype activations...'.format(hub.mode, context_type))
    probe_prototype_acts = []
    for n, probe_x_mat in enumerate(hub.probe_x_mats):
        # x
        if context_type == 'none':
            x = probe_x_mat[:, -1][:, np.newaxis]
        elif context_type == 'ordered':
            x = probe_x_mat
        elif context_type == 'x':
            x = probe_x_mat[:, :-1]
        elif context_type == 'last':
            x = probe_x_mat[:, -2][:, np.newaxis]
        elif context_type == 'shuffled':
            x_no_probe = probe_x_mat[:, np.random.permutation(np.arange(probe_x_mat.shape[1] - 1))]
            x = np.hstack((x_no_probe, np.expand_dims(probe_x_mat[:, -1], axis=1)))
        else:
            raise AttributeError('starting_small: Invalid arg to "context_type".')
        # probe_act
        probe_exemplar_acts_mat = sess.run(h, feed_dict={graph.x: x})
        probe_prototype_act = np.mean(probe_exemplar_acts_mat, axis=0)
        probe_prototype_acts.append(probe_prototype_act)
    res = np.vstack(probe_prototype_acts)
    return res


def make_gold(hub):
    num_rows = hub.probe_store.num_probes
    num_cols = hub.probe_store.num_probes
    res = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        probe1 = hub.probe_store.types[i]
        for j in range(num_cols):
            probe2 = hub.probe_store.types[j]
            if hub.probe_store.probe_cat_dict[probe1] == hub.probe_store.probe_cat_dict[probe2]:
                res[i, j] = 1
    return res.astype(np.bool)


def calc_cluster_score(hub, probe_sims, cluster_metric):
    print('Computing {} score...'.format(cluster_metric))

    def calc_signals(probe_sims, gold, thr):  # vectorized algorithm is 20X faster
        predicted = np.zeros_like(probe_sims, int)
        predicted[np.where(probe_sims > thr)] = 1
        tp = float(len(np.where((predicted == gold) & (gold == 1))[0]))
        tn = float(len(np.where((predicted == gold) & (gold == 0))[0]))
        fp = float(len(np.where((predicted != gold) & (gold == 0))[0]))
        fn = float(len(np.where((predicted != gold) & (gold == 1))[0]))
        return tp, tn, fp, fn

    def calc_probes_fs(thr):  # TODO is wrong - tensorflow shows increasing f1 but this never does
        tp, tn, fp, fn = calc_signals(thr)
        precision = np.divide(tp, (tp + fp))
        sensitivity = np.divide(tp, (tp + fn))  # aka recall

        fs = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-7)  # TODO test

        return fs

    def calc_probes_ck(thr):
        """
        cohen's kappa
        """
        tp, tn, fp, fn = calc_signals(thr)
        totA = np.divide(tp + tn, (tp + tn + fp + fn))
        #
        pyes = ((tp + fp) / (tp + fp + tn + fn)) * ((tp + fn) / (tp + fp + tn + fn))
        pno = ((fn + tn) / (tp + fp + tn + fn)) * ((fp + tn) / (tp + fp + tn + fn))
        #
        randA = pyes + pno
        ck = (totA - randA) / (1 - randA)
        # print('totA={:.2f} randA={:.2f}'.format(totA, randA))
        return ck

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals(thr)
        specificity = np.divide(tn + 1e-10, (tn + fp + 1e-10))
        sensitivity = np.divide(tp + 1e-10, (tp + fn + 1e-10))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    gold = make_gold(hub)
    calc_signals = partial(calc_signals, probe_sims, gold)
    sims_mean = np.mean(probe_sims).item()
    print('Finding best threshold using bayesian-optimization...')
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    if cluster_metric == 'f1':
        fun = calc_probes_fs
    elif cluster_metric == 'ba':
        fun = calc_probes_ba
    elif cluster_metric == 'ck':
        fun = calc_probes_ck
    else:
        raise AttributeError('rnnlab: Invalid arg to "metric".')
    bo = BayesianOptimization(fun, {'thr': (-1.0, 1.0)}, verbose=False)
    bo.explore(
        {'thr': [sims_mean]})  # keep bayes-opt at version 0.6 because 1.0 occasionally returns 0.50 wrongly
    bo.maximize(init_points=2, n_iter=config.Eval.num_opt_steps,
                acq="poi", xi=0.01, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = fun(best_thr)
    res = np.mean(results)
    return res


def calc_h_term_sims(hub, graph, sess, h):
    print('Making h_term_sims...')
    term_h_acts_sum = np.zeros((hub.params.num_types, hub.params.embed_size))
    # collect acts
    pbar = pyprind.ProgBar(hub.num_mbs_in_token_ids)
    for (x, y) in hub.gen_ids(num_iterations=1):
        pbar.update()
        acts_mat = sess.run(h, feed_dict={graph.x: x, graph.y: y})
        # update term_h_acts_sum
        last_term_ids = [term_id for term_id in x[:, -1]]
        for term_id, acts in zip(last_term_ids, acts_mat):
            term_h_acts_sum[term_id] += acts
    # term_h_acts
    term_h_acts = np.zeros_like(term_h_acts_sum)
    for term, term_id in hub.train_terms.term_id_dict.items():
        term_freq = hub.train_terms.term_freq_dict[term]
        term_h_acts[term_id] = term_h_acts_sum[term_id] / max(1.0, term_freq)
    # term_h_sims
    res = cosine_similarity(term_h_acts)
    # nan check
    check_nans(term_h_acts, 'term_h_acts')
    check_nans(res, 'term_h_sims')
    return res


def calc_pp(hub, graph, sess, is_test):
    print('Calculating {} perplexity...'.format('test' if is_test else 'train'))
    pp_sum, num_batches, pp = 0, 0, 0
    pbar = pyprind.ProgBar(hub.num_mbs_in_token_ids)
    for (x, y) in hub.gen_ids(num_iterations=1, is_test=is_test):
        pbar.update()
        pp_batch = sess.run(graph.mean_pp, feed_dict={graph.x: x, graph.y: y})
        pp_sum += pp_batch
        num_batches += 1
    pp = pp_sum / num_batches
    print('pp={}'.format(pp))
    return pp