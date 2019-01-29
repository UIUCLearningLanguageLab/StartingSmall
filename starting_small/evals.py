import pyprind
import numpy as np
from bayes_opt import BayesianOptimization
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity

from starting_small import config
from starting_small.evalutils import sample_from_iterable


def check_nans(mat, name='mat'):
    if np.any(np.isnan(mat)):
        num_nans = np.sum(np.isnan(mat))
        print('Found {} Nans in {}'.format(num_nans, name), 'red')


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

    def calc_signals(_probe_sims, _labels, thr):  # vectorized algorithm is 20X faster
        probe_sims_clipped = np.clip(_probe_sims, 0, 1)
        probe_sims_clipped_triu = probe_sims_clipped[np.triu_indices(len(probe_sims_clipped), k=1)]
        predictions = np.zeros_like(probe_sims_clipped_triu, int)
        predictions[np.where(probe_sims_clipped_triu > thr)] = 1
        #
        tp = float(len(np.where((predictions == _labels) & (_labels == 1))[0]))
        tn = float(len(np.where((predictions == _labels) & (_labels == 0))[0]))
        fp = float(len(np.where((predictions != _labels) & (_labels == 0))[0]))
        fn = float(len(np.where((predictions != _labels) & (_labels == 1))[0]))
        return tp, tn, fp, fn

    # define calc_signals_partial
    check_nans(probe_sims, name='probe_sims')
    gold_mat = make_gold(hub)
    labels = gold_mat[np.triu_indices(len(gold_mat), k=1)]
    calc_signals_partial = partial(calc_signals, probe_sims, labels)

    def calc_probes_fs(thr):
        """
        WARNING: this gives incorrect results at early timepoints (lower compared to tensorflow implementation)
        # TODO this may be due to using sim_mean as first point to bayesian-opt:
        # TODO sim mean might not be good init point for f-score (but it is for ba)

        """
        tp, tn, fp, fn = calc_signals_partial(thr)
        precision = np.divide(tp + 1e-7, (tp + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        fs = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-7)
        return fs

    def calc_probes_ck(thr):
        """
        cohen's kappa
        """
        tp, tn, fp, fn = calc_signals_partial(thr)
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
        tp, tn, fp, fn = calc_signals_partial(thr)
        specificity = np.divide(tn + 1e-7, (tn + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    sims_mean = np.mean(probe_sims).item()
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    if cluster_metric == 'f1':
        fun = calc_probes_fs
    elif cluster_metric == 'ba':
        fun = calc_probes_ba
    elif cluster_metric == 'ck':
        fun = calc_probes_ck
    else:
        raise AttributeError('rnnlab: Invalid arg to "cluster_metric".')
    bo = BayesianOptimization(fun, {'thr': (0.0, 1.0)}, verbose=False)
    bo.explore(
        {'thr': [0.99, sims_mean]})  # keep bayes-opt at version 0.6 because 1.0 occasionally returns 0.50 wrongly
    bo.maximize(init_points=config.Eval.num_opt_init_steps, n_iter=config.Eval.num_opt_steps,
                acq="poi", xi=0.01, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = fun(best_thr)
    res = np.mean(results)
    return res


def adjust_context(mat, context_type):
    if context_type == 'none':
        x = mat[:, -1][:, np.newaxis]
    elif context_type == 'ordered':
        x = mat
    elif context_type == 'x':
        x = mat[:, :-1]
    elif context_type == 'last':
        x = mat[:, -2][:, np.newaxis]
    elif context_type == 'shuffled':
        x_no_probe = mat[:, np.random.permutation(np.arange(mat.shape[1] - 1))]
        x = np.hstack((x_no_probe, np.expand_dims(mat[:, -1], axis=1)))
    else:
        raise AttributeError('starting_small: Invalid arg to "context_type".')
    return x


def make_probe_prototype_acts_mat(hub, context_type, graph, sess, h):
    print('Making "{}" "{}" probe prototype activations...'.format(hub.mode, context_type))

    # TODO debug
    for k, v in hub.params.__dict__.items():
        print(k, v)


    res = np.zeros((hub.probe_store.num_probes, hub.params.embed_size))
    for n, probe_x_mat in enumerate(hub.probe_x_mats):
        x = adjust_context(probe_x_mat, context_type)
        # probe_act
        probe_exemplar_acts_mat = sess.run(h, feed_dict={graph.x: x})
        probe_prototype_act = np.mean(probe_exemplar_acts_mat, axis=0)
        res[n] = probe_prototype_act
    return res


def calc_h_term_sims(hub, context_type, graph, sess, h):
    print('Making "{}" "{}" h_term_sims...'.format(hub.mode, context_type))
    term_h_acts_sum = np.zeros((hub.params.num_types, hub.params.embed_size))
    # collect acts
    # pbar = pyprind.ProgBar(hub.num_mbs_in_token_ids)
    pbar = pyprind.ProgBar(config.Eval.num_h_samples)
    num_samples = 0
    num_iterations_list = [1] * hub.params.num_parts
    for (x, y) in sample_from_iterable(hub.gen_ids(num_iterations_list), config.Eval.num_h_samples):
        x = adjust_context(x, context_type)  # TODO test here (works in make_probe_prototype_acts_mat)
        acts_mat = sess.run(h, feed_dict={graph.x: x, graph.y: y})
        # update term_h_acts_sum
        last_term_ids = [term_id for term_id in x[:, -1]]
        for term_id, acts in zip(last_term_ids, acts_mat):
            term_h_acts_sum[term_id] += acts
        num_samples += 1
        pbar.update()
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