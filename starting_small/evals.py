import pyprind
import numpy as np
from bayes_opt import BayesianOptimization
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score

from starting_small import config
from starting_small.evalutils import sample_from_iterable


def calc_w_term_sims(hub, graph, sess, word_type, w_name):
    """
    overall similarity of weights (wx or wy)
    similarity of weights starts of at zero and increase gradually,
    contrary to similarity of contextualized hidden states which jumps to near 1.0 at first timepoint
    """
    if word_type == 'probes':
        term_ids = [hub.train_terms.term_id_dict[probe] for probe in hub.probe_store.types]
    elif word_type == 'nouns':
        term_ids = [hub.train_terms.term_id_dict[noun] for noun in hub.nouns]
    elif word_type == 'terms':
        np.random.seed(1)
        term_ids = np.arange(0, hub.params.num_types // 2)  # divide by 2 to save memory
    else:
        raise AttributeError('Invalid arg to "word_type": "{}"'.format(word_type))
    #
    if w_name == 'wx':
        w = sess.run(graph.wx)
        w_filtered = w[term_ids, :]
    elif w_name == 'wy':
        w = sess.run(graph.wy)
        w_filtered = w[:, term_ids].T
    else:
        raise AttributeError('rnnlab: Invalid arg to "w_name"')
    #
    sims = cosine_similarity(w_filtered, w_filtered).mean(axis=1)

    # TODO save memory by returning only upper triangle?  upper_triang = mat[np.triu_indices(len(mat), k=1)]

    print('{} {} mean_sim={}'.format(word_type, w_name, sims.mean()))
    return sims


def calc_pos_map(hub, graph, sess, pos, max_x=2 ** 16, max_term_windows=16):
    """
    return mean-average-precision between correct terms (members of "pos") and predictions (softmax probabilities)
    """
    # calc softmax
    cat_terms = getattr(hub, pos)
    windows_list = [hub.get_term_id_windows(cat_term, num_samples=max_term_windows)
                    for cat_term in cat_terms]
    windows = [window for windows in windows_list for window in windows]
    if not windows:
        raise RuntimeError('Did not find windows for POS "{}"'.format(pos))
    x = np.vstack(windows)[:max_x, :-1]  # -1 because last term is predicted
    feed_dict = {graph.x: x}
    y_pred_mat = sess.run(graph.softmax_probs, feed_dict=feed_dict)
    # score
    cat_term_ids = [hub.train_terms.term_id_dict[term] for term in cat_terms]
    y_true = np.zeros(hub.params.num_types)
    y_true[cat_term_ids] = 1
    result = np.mean([average_precision_score(y_true, y_pred) for y_pred in y_pred_mat])
    return result


def check_nans(mat, name='mat'):
    if np.any(np.isnan(mat)):
        num_nans = np.sum(np.isnan(mat))
        print('Found {} Nans in {}'.format(num_nans, name))


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
        # TODO this not due to using sim_mean as first point to bayesian-opt:
        # TODO perhaps exploration-exploitation settings are only good for ba but not f1

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
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}  # without this, warnings about predicted variance < 0
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
    print(res)
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
        x = adjust_context(x, context_type)
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
    num_iterations_list = [1] * hub.params.num_parts
    for (x, y) in hub.gen_ids(num_iterations_list, is_test=is_test):
        pbar.update()
        pp_batch = sess.run(graph.batch_pp, feed_dict={graph.x: x, graph.y: y})
        pp_sum += pp_batch
        num_batches += 1
    pp = pp_sum / num_batches
    print('pp={}'.format(pp))
    return pp