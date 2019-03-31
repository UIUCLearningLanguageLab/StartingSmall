from cytoolz import itertoolz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

from analysis.hierarchical_data_utils import make_data, make_probe_data, calc_cluster_score
from analysis.rnn import RNN


NUM_TOKENS = 1 * 10 ** 5  # TODO 6 might overload RAM
MAX_NGRAM_SIZE = 1
NUM_DESCENDANTS = 2  # use as num_cats
NUM_LEVELS = 7
E = 0.2

MB_SIZE = 64
LEARNING_RATE = (0.01, 0.00, 20)
NUM_EPOCHS = 1
NUM_HIDDENS = 256

NUM_CATS = 10
NUM_CAT_MEMBERS = 5
THRESHOLDS = [10, 20]  # TODO
NGRAM_SIZE_FOR_CAT = 1  # TODO manipulate this - or concatenate all structures?
MIN_PROBE_FREQ = 1
SENTENCE_LEN = 7  # TODO this destroys hierarchical structure but is necessary to ensure all types occur in tokens

# make tokens with hierarchical n-gram structure
vocab, tokens, token_ids, word2id, ngram2structure_mat = make_data(
    NUM_TOKENS, MAX_NGRAM_SIZE, NUM_DESCENDANTS, NUM_LEVELS, E, sentence_len=SENTENCE_LEN)
num_vocab = len(vocab)
num_types_in_tokens = len(set(tokens))
print()
print('num_vocab={}'.format(num_vocab))
print('num types in tokens={}'.format(num_types_in_tokens))
if not num_types_in_tokens == num_vocab:
    raise RuntimeError('Not all types were found in tokens.'
                       'Decrease NUM_LEVELS, increase NUM_TOKENS, or decrease SENTENCE_LEN.')

#
num_theoretical_legals = num_vocab / (2 ** MAX_NGRAM_SIZE)
print('num_theoretical_legals={}'.format(num_theoretical_legals))  # perplexity should converge to this value
# TODO pp only converges to num_theoretical_legals because all other types occur only rarely due to random process
# TODO involved in breaking tokens into sentences (SENTENCE_LEN)

# train_seqs
train_seqs = []
for seq in itertoolz.partition_all(MB_SIZE, token_ids):  # a seq contains 64 token_ids
    if len(seq) == MB_SIZE:
        train_seqs.append(list(seq))  # need to convert tuple to list
print('num sequences={}'.format(len(train_seqs)))

# train + eval
for thr in THRESHOLDS:
    # categories
    print('Making categories with thr={}'.format(thr))
    structure_mat = ngram2structure_mat[NGRAM_SIZE_FOR_CAT]  # TODO concatenate all
    probes, probe2cat = make_probe_data(structure_mat, vocab, NUM_CATS, NUM_CAT_MEMBERS, thr)
    c = Counter(tokens)
    for p in probes:
        assert c[p] > MIN_PROBE_FREQ
    print('Collected {} probes'.format(len(probes)))
    # check probe sim
    assert len(structure_mat) == len(word2id)
    probe_acts = structure_mat[:, [word2id[p] for p in probes]].T  # TODO  why cols but not rows give above-50 ba?
    # probe_acts = structure_mat[[word2id[p] for p in probes], :]
    print('probe_acts', probe_acts.shape)
    probe_sims = cosine_similarity(probe_acts)
    ba = calc_cluster_score(probe_sims, probes, probe2cat)
    print('ba={}'.format(ba))
    # train srn
    srn = RNN(input_size=num_vocab,
              learning_rate=LEARNING_RATE,
              num_epochs=NUM_EPOCHS,
              num_hiddens=NUM_HIDDENS,
              num_seqs_in_batch=1)  # num_seqs_in_batch must be 1
    srn.train(train_seqs, verbose=False)  # TODO this does not train on GPU
    # calc ba
    print('Getting probe activations...')
    wx = srn.retrieve_wx_for_analysis()  # TODO retrieve hidden states instead?
    probe_acts = np.asarray([wx[word2id[p], :] for p in probes])
    print('probe_acts', probe_acts.shape)
    assert len(wx) == num_vocab
    assert len(probe_acts) == len(probes)
    probe_sims = cosine_similarity(probe_acts)
    ba = calc_cluster_score(probe_sims, probes, probe2cat)
    print('ba={}'.format(ba))
    print()


