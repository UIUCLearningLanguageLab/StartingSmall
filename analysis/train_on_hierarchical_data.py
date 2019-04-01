from cytoolz import itertoolz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.hierarchical_data_utils import make_data, make_probe_data, calc_ba
from analysis.rnn import RNN


NUM_TOKENS = 1 * 10 ** 5
MAX_NGRAM_SIZE = 1
NUM_DESCENDANTS = 2  # 2
NUM_LEVELS = 9  # 12
E = 0.2  # 0.2

MB_SIZE = 64
LEARNING_RATE = (0.1, 0.00, 20)
NUM_EPOCHS = 20  # 10
NUM_HIDDENS = 512
BPTT = MAX_NGRAM_SIZE
CALC_PP = True  # must set to False if train_seqs to big to calc pp in one batch

NUM_CATS = 2
NUM_CAT_MEMBERS = 50
THRESHOLDS = [200, 50]
NGRAM_SIZE_FOR_CAT = 1  # TODO manipulate this - or concatenate all structures?
MIN_PROBE_FREQ = 5


def plot_ba_trajs(d):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=None)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0.5, 1.0])
    # plot
    num_summaries = len(d)
    palette = iter(sns.color_palette('hls', num_summaries))
    for thr, bas in d.items():
        ax.plot(bas, '-', color=next(palette),
                label='thr={}'.format(thr))
    plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=1.0, frameon=False)
    plt.tight_layout()
    plt.show()


# make tokens with hierarchical n-gram structure
vocab, tokens, token_ids, word2id, ngram2legals_mat = make_data(
    NUM_TOKENS, MAX_NGRAM_SIZE, NUM_DESCENDANTS, NUM_LEVELS, E)
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

# train_seqs
train_seqs = []
for seq in itertoolz.partition_all(MB_SIZE, token_ids):  # a seq contains MB_SIZE token_ids
    if len(seq) == MB_SIZE:
        train_seqs.append(list(seq))  # need to convert tuple to list
print('num sequences={}'.format(len(train_seqs)))

# train + eval
thr2bas = {thr: [] for thr in THRESHOLDS}
for thr in THRESHOLDS:
    # categories
    print('Making {} categories with num_members={} and thr={}...'.format(NUM_CATS, NUM_CAT_MEMBERS, thr))
    legals_mat = ngram2legals_mat[NGRAM_SIZE_FOR_CAT]
    probes, probe2cat = make_probe_data(legals_mat, vocab, NUM_CATS, NUM_CAT_MEMBERS, thr, verbose=False, plot=True)
    c = Counter(tokens)
    for p in probes:
        # print('"{:<10}" {:>4}'.format(p, c[p]))  # check for bimodality
        assert c[p] > MIN_PROBE_FREQ
    print('Collected {} probes'.format(len(probes)))
    # check probe sim
    probe_acts1 = legals_mat[[word2id[p] for p in probes], :]
    ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
    probe_acts2 = legals_mat[:, [word2id[p] for p in probes]].T
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
        pp = srn.calc_seqs_pp(train_seqs[:10]) if CALC_PP else 0
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
        thr2bas[thr].append(ba)
    print()

plot_ba_trajs(thr2bas)