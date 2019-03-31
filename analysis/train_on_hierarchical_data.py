from cytoolz import itertoolz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

from analysis.hierarchical_data_utils import make_data, make_probe_data, calc_ba
from analysis.rnn import RNN


NUM_TOKENS = 1 * 10 ** 5  # TODO 6 might overload RAM
MAX_NGRAM_SIZE = 1
NUM_DESCENDANTS = 2  # use as num_cats
NUM_LEVELS = 8
E = 0.2

MB_SIZE = 64
LEARNING_RATE = (0.01, 0.00, 20)
NUM_EPOCHS = 10
NUM_HIDDENS = 128
BPTT = MAX_NGRAM_SIZE
CALC_PP = False

NUM_CATS = 10
NUM_CAT_MEMBERS = 5
THRESHOLDS = [5, 10]
NGRAM_SIZE_FOR_CAT = 1  # TODO manipulate this - or concatenate all structures?
MIN_PROBE_FREQ = 1
SENTENCE_LEN = 64  # TODO this destroys hierarchical structure but is necessary to ensure all types occur in tokens


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
    structure_mat = ngram2structure_mat[NGRAM_SIZE_FOR_CAT]  # TODO concatenate all?
    probes, probe2cat = make_probe_data(structure_mat, vocab, NUM_CATS, NUM_CAT_MEMBERS, thr, verbose=True)
    c = Counter(tokens)
    for p in probes:
        assert c[p] > MIN_PROBE_FREQ
    print('Collected {} probes'.format(len(probes)))
    # check probe sim
    assert len(structure_mat) == len(word2id)
    probe_acts1 = structure_mat[[word2id[p] for p in probes], :]
    ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
    probe_acts2 = structure_mat[:, [word2id[p] for p in probes]].T
    ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
    print('input-data row-wise ba={:.3f}'.format(ba1))
    print('input-data col-wise ba={:.3f}'.format(ba2))
    print()
    # srn
    srn = RNN(input_size=num_vocab,
              learning_rate=LEARNING_RATE,
              num_epochs=NUM_EPOCHS,
              num_hiddens=NUM_HIDDENS,
              bptt=BPTT,
              num_seqs_in_batch=1)  # num_seqs_in_batch must be 1
    # train + evaluate
    lr = srn.learning_rate[0]  # initial
    decay = srn.learning_rate[1]
    num_epochs_without_decay = srn.learning_rate[2]
    for epoch in range(srn.num_epochs):
        # perplexity
        pp = srn.calc_seqs_pp(train_seqs) if CALC_PP else 0
        # ba
        wx = srn.get_wx()  # TODO retrieve hidden states instead?
        p_acts = np.asarray([wx[word2id[p], :] for p in probes])
        ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
        # train
        lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
        lr = lr * lr_decay  # decay lr if it is time
        srn.train_epoch(train_seqs, lr, verbose=False)
        #
        print('epoch={:>2}/{:>2} | pp={:>5} ba={:.3f}'.format(epoch, srn.num_epochs, int(pp), ba))
    print()


