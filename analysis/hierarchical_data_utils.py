import numpy as np
import pyprind
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from bayes_opt import BayesianOptimization
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing as mp


def generate_tokens_from_zipfian(vocab, num_tokens):  # TODO use
    num_vocab = len(vocab)
    res = [vocab[i] if i < num_vocab else 'OOV' for i in np.random.zipf(2, num_tokens)]
    return res


def cluster(data_mat, original_row_words=None, original_col_words=None):
    print('Clustering...')
    #
    lnk0 = linkage(pdist(data_mat))
    dg0 = dendrogram(lnk0,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)
    z = data_mat[dg0['leaves'], :]  # reorder rows
    #
    lnk1 = linkage(pdist(data_mat.T))
    dg1 = dendrogram(lnk1,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)

    z = z[:, dg1['leaves']]  # reorder cols
    #
    if original_row_words is None and original_col_words is None:
        return z
    else:
        row_labels = np.array(original_row_words)[dg0['leaves']]
        col_labels = np.array(original_col_words)[dg1['leaves']]
        return z, row_labels, col_labels


def make_probe_data(data_mat, vocab, num_cats, num_members, min_count,
                    method='average', metric='cityblock', verbose=False, plot=True):
    """
    make categories from hierarchically organized data.
    """
    num_vocab = len(vocab)
    assert data_mat.shape == (num_vocab, num_vocab)
    # linkage will return an array of length n - 1
    # giving you information about the n - 1 cluster merges which it needs to pairwise merge n clusters.
    # Z[i] will tell us which clusters were merged in the i-th iteration
    z = linkage(data_mat, method, metric)
    c, _ = cophenet(z, pdist(data_mat, metric))
    assert c > 0.8

    def get_all_probes_in_tree(res, visited, z, node_id):
        try:
            p = vocab[node_id.astype(int)]
        except IndexError:  # idx does not refer to leaf node (it refers to cluster)
            new_node_id1 = z[node_id.astype(int) - len(vocab)][0]
            new_node_id2 = z[node_id.astype(int) - len(vocab)][1]
            if new_node_id1 not in visited:
                get_all_probes_in_tree(res, visited, z, new_node_id1)
            if new_node_id2 not in visited:
                get_all_probes_in_tree(res, visited, z, new_node_id2)
        else:
            res.append(p)
            visited.append(node_id)

    # define categories
    assert num_members <= min_count
    probes = []
    probe2cat = {}
    cat_id = 0
    visited_node_ids = []  # prevents descending the same tree more than once (otherwise cats share probes)
    for row in z:
        if cat_id == num_cats:
            break
        idx1, idx2, dist, count = row  # idx >= len(X) actually refer to the cluster formed in Z[idx - len(X)]
        #
        if count >= min_count:
            unvisited_ps = []
            get_all_probes_in_tree(unvisited_ps, visited_node_ids, z, idx1)  # populates probes_in_cluster
            get_all_probes_in_tree(unvisited_ps, visited_node_ids, z, idx2)  # populates probes_in_cluster
            if len(unvisited_ps) < min_count:
                print('WARNING: Found cluster which includes nodes which previously have been visited.')
                # raise RuntimeError('Cluster {} includes nodes which previously have been visited'.format(cat_id))
            if len(unvisited_ps) < num_members:
                print('WARNING: Found cluster with unvisited nodes < num_members - skipping.')
                continue  # TODO just increment cat_id instead?
            #
            probe_cats = unvisited_ps[-num_members:]  # get last because they are on lowest levels of tree
            probes.extend(probe_cats)
            probe2cat.update({p: cat_id for p in probe_cats})
            cat_id += 1
            if verbose:
                print('cluster {}: num unvisited nodes={} | using {} as probes'.format(
                    cat_id, len(unvisited_ps), len(probe_cats)))
                print()
    else:
        raise RuntimeError('Too many categories. Could not make all categories from given data. ')

    if plot:
        annotate_above = num_vocab
        fig, ax = plt.subplots(figsize=(40, 10), dpi=200)
        ddata = dendrogram(z, ax=ax, leaf_rotation=90., leaf_font_size=12,)
        # label only probes
        reordered_vocab = np.asarray(vocab)[ddata['leaves']]
        ax.set_xticklabels([w if w in probes else '' for w in reordered_vocab], fontsize=5)
        plt.title('Hierarchical Clustering Dendrogram\n'
                  'num_cats={}, num_members={}, min_count={} '.format(
            num_cats, num_members, min_count), fontsize=20)
        plt.xlabel('words in vocab (only probes are shown)')
        plt.ylabel('{} distance'.format(metric))
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate('{}'.format(int(y)), (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        plt.show()

    # assert len(probes) == num_members * num_cats
    return probes, probe2cat


def sample_from_hierarchical_diffusion(node0, num_descendants, num_levels, e):
    """the higher the change probability (e),
     the less variance accounted for by more distant nodes in diffusion process"""
    nodes = [node0]
    for level in range(num_levels):
        candidate_nodes = nodes * num_descendants
        nodes = [node if p else -node for node, p in zip(candidate_nodes,
                                                         np.random.binomial(n=2, p=1 - e, size=len(candidate_nodes)))]
    return nodes


def make_chunk(chunk_id, size2word2legals_, vocab_, num_start_, chunk_size_, random_interval=np.nan):
    tokens_chunk = np.random.choice(vocab_, size=num_start_).tolist()  # prevents indexError at start
    pbar = pyprind.ProgBar(chunk_size_) if chunk_id == 0 else None
    for loc in range(chunk_size_):
        # append random word to break structure into pseudo-sentences
        if loc % random_interval == 0:
            new_token = np.random.choice(vocab_, size=1).item()
            tokens_chunk.append(new_token)
            continue
        # append word which is constrained by hierarchical structure
        else:
            # get words which are legal to come next
            legals = set(vocab_)
            for size, word2legals in size2word2legals_.items():
                previous_token = tokens_chunk[-size]
                legals.intersection_update(word2legals[previous_token])
            # sample uniformly from legals
            try:
                new_token = np.random.choice(list(legals), size=1).item()
            except ValueError:  # no legals
                raise RuntimeError('No legal next word available.'
                                   'Increase E - the probability of a flip in hierarchical diffusion process')
            # collect
            tokens_chunk.append(new_token)
        pbar.update() if chunk_id == 0 else None
    return tokens_chunk


def make_data(num_tokens, max_ngram_size=6, num_descendants=2, num_levels=12, e=0.01,
              num_chunks=4):
    """
    generate text by adding one word at a time to a list of words.
    each word is constrained by the legals matrices - which are hierarchical -
    and determine the legal successors for each word in the vocabulary.
    there is one legal matrix for each ngram (in other words, for each distance up to MAX_NGRAM_SIZE)
    the set of legal words that can follow a word is the intersection of legal words dictated by each legals matrix.
    the size of the intersection is approximately the same for each word, and a word is sampled uniformly from this set
    the size of the intersection is the best possible perplexity a model can achieve,
    because perplexity indicates the number of choices from a random uniformly distributed set of choices
    """
    # vocab
    num_vocab = num_descendants ** num_levels
    vocab = ['w{}'.format(i) for i in np.arange(num_vocab)]
    # ngram2legals_mat - each row specifies legal next words (col_words)
    ngram_sizes = range(1, max_ngram_size + 1)
    word2node0 = {}
    ngram2slegals_mat = {ngram: np.zeros((num_vocab, num_vocab), dtype=np.int) for ngram in ngram_sizes}
    print('Making hierarchical dependency structure...')
    for ngram_size in ngram_sizes:
        for row_id, word in enumerate(vocab):
            node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
            word2node0[word] = node0
            ngram2slegals_mat[ngram_size][row_id, :] = sample_from_hierarchical_diffusion(
                node0, num_descendants, num_levels, e)
    print('Done')
    # collect legal next words for each word at each distance - do this once to speed calculation of tokens
    # whether -1 or 1 determines legality depends on node0 - otherwise half of words are never legal
    size2word2legals = {}
    for ngram_size in ngram_sizes:
        legals_mat = ngram2slegals_mat[ngram_size]  # this must be transposed
        word2legals = {}
        for row_word, row in zip(vocab, legals_mat):
            word2legals[row_word] = [w for w, val in zip(vocab, row) if val == word2node0[w]]
        size2word2legals[ngram_size] = word2legals
    # get one token at a time
    pool = mp.Pool(processes=num_chunks)
    chunk_size = num_tokens // num_chunks
    results = [pool.apply_async(make_chunk, args=(chunk_id, size2word2legals, vocab, max_ngram_size, chunk_size))
               for chunk_id in range(num_chunks)]
    tokens = []
    print('Creating tokens from hierarchical dependency structure...')
    try:
        for res in results:
            tokens += res.get()
        pool.close()
    except KeyboardInterrupt:
        pool.close()
        raise SystemExit('Interrupt occurred during multiprocessing. Closed worker pool.')
    print('Done')
    return vocab, tokens, ngram2slegals_mat


def calc_ba(probe_sims, probes, probe2cat, num_opt_init_steps=1, num_opt_steps=10):
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

    # gold_mat
    if not len(probes) == probe_sims.shape[0] == probe_sims.shape[1]:
        raise RuntimeError(len(probes), probe_sims.shape[0], probe_sims.shape[1])
    num_rows = len(probes)
    num_cols = len(probes)
    gold_mat = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        probe1 = probes[i]
        for j in range(num_cols):
            probe2 = probes[j]
            if probe2cat[probe1] == probe2cat[probe2]:
                gold_mat[i, j] = 1

    # define calc_signals_partial
    labels = gold_mat[np.triu_indices(len(gold_mat), k=1)]
    calc_signals_partial = partial(calc_signals, probe_sims, labels)

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals_partial(thr)
        specificity = np.divide(tn + 1e-7, (tn + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    sims_mean = np.mean(probe_sims).item()
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}  # without this, warnings about predicted variance < 0
    bo = BayesianOptimization(calc_probes_ba, {'thr': (0.0, 1.0)}, verbose=False)
    bo.explore(
        {'thr': [sims_mean]})  # keep bayes-opt at version 0.6 because 1.0 occasionally returns 0.50 wrongly
    bo.maximize(init_points=num_opt_init_steps, n_iter=num_opt_steps,
                acq="poi", xi=0.01, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = calc_probes_ba(best_thr)
    res = np.mean(results)
    return res
