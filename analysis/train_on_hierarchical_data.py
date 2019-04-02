from cytoolz import itertoolz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.hierarchical_data_utils import make_data, make_probe_data, calc_ba
from analysis.rnn import RNN


NUM_TOKENS = 5 * 10 ** 6  # must always be at least 1M
MAX_NGRAM_SIZE = 1
NUM_DESCENDANTS = 2  # 2
NUM_LEVELS = 8  # 12
E = 0.2  # 0.2
ZIPF_A = 2  # TODO test

MB_SIZE = 64
LEARNING_RATE = (0.001, 0.00, 20)  # 0.01 is too fast  # TODO
NUM_EPOCHS = 20
NUM_HIDDENS = 128
BPTT = MAX_NGRAM_SIZE
NUM_PP_SEQS = 10  # number of documents to calc perplexity for

PARENT_COUNT = 256  # exact size of single parent cluster
NUM_CATS_LIST = [2, 4, 8, 16, 32]
NGRAM_SIZE_FOR_CAT = 1  # TODO manipulate this - or concatenate all structures?
MIN_PROBE_FREQ = 10


def plot_ba_trajs(d1, d2, title):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=None)
    plt.title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim([0.5, 1.0])
    # plot
    num_trajs = len(d1)
    palette = iter(sns.color_palette('hls', num_trajs))
    for num_cats, bas in sorted(d1.items(), key=lambda i: i[0]):
        c = next(palette)
        ax.plot(bas, '-', color=c,
                label='num_cats={}'.format(num_cats))
        ax.axhline(y=d2[num_cats], linestyle='dashed', color=c)
    plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=1.0, frameon=False)
    plt.tight_layout()
    plt.show()


# make tokens with hierarchical n-gram structure
vocab, tokens, ngram2legals_mat = make_data(
    ZIPF_A, NUM_TOKENS, MAX_NGRAM_SIZE, NUM_DESCENDANTS, NUM_LEVELS, E)
num_vocab = len(vocab)
num_types_in_tokens = len(set(tokens))
word2id = {word: n for n, word in enumerate(vocab)}
token_ids = [word2id[w] for w in tokens]
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


# probes_data
num_cats2probes_data = {}
num_cats2max_ba = {}
for num_cats in NUM_CATS_LIST:
    print('Getting {} categories with MIN_COUNT={}...'.format(num_cats, PARENT_COUNT))
    legals_mat = ngram2legals_mat[NGRAM_SIZE_FOR_CAT]
    probes, probe2cat = make_probe_data(legals_mat, vocab, num_cats, PARENT_COUNT,
                                        plot=False)
    num_cats2probes_data[num_cats] = (probes, probe2cat)
    c = Counter(tokens)
    for p in probes:
        # print('"{:<10}" {:>4}'.format(p, c[p]))  # check for bi-modality
        if c[p] < MIN_PROBE_FREQ:
            print('WARNING: "{}" occurs only {} times'.format(p, c[p]))
    print('Collected {} probes'.format(len(probes)))
    # check probe sim
    probe_acts1 = legals_mat[[word2id[p] for p in probes], :]
    ba1 = calc_ba(cosine_similarity(probe_acts1), probes, probe2cat)
    probe_acts2 = legals_mat[:, [word2id[p] for p in probes]].T
    ba2 = calc_ba(cosine_similarity(probe_acts2), probes, probe2cat)
    print('input-data row-wise ba={:.3f}'.format(ba1))
    print('input-data col-wise ba={:.3f}'.format(ba2))
    print()
    num_cats2max_ba[num_cats] = ba2

# srn
srn = RNN(input_size=num_vocab,
          rnn_type='srn',
          num_hiddens=NUM_HIDDENS,
          num_epochs=NUM_EPOCHS,
          learning_rate=LEARNING_RATE,
          optimization='adagrad',
          bptt=BPTT,
          num_seqs_in_batch=1)  # num_seqs_in_batch must be 1
# train once + evaluate on all category structures
lr = srn.learning_rate[0]  # initial
decay = srn.learning_rate[1]
num_epochs_without_decay = srn.learning_rate[2]
num_cats2bas = {num_cats: [] for num_cats in NUM_CATS_LIST}
for epoch in range(srn.num_epochs):
    # perplexity
    pp = srn.calc_seqs_pp(train_seqs[:NUM_PP_SEQS])  # TODO calc pp on all seqs
    # ba
    for num_cats, (probes, probe2cat) in sorted(num_cats2probes_data.items(), key=lambda i: i[0]):
        wx = srn.get_wx()  # TODO also test wy
        p_acts = np.asarray([wx[word2id[p], :] for p in probes])
        ba = calc_ba(cosine_similarity(p_acts), probes, probe2cat)
        num_cats2bas[num_cats].append(ba)
        print('epoch={:>2}/{:>2} | ba={:.3f} num_cats={}'.format(epoch, srn.num_epochs, ba, num_cats))
    # train
    lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
    lr = lr * lr_decay  # decay lr if it is time
    srn.train_epoch(train_seqs, lr, verbose=False)
    #
    print('epoch={:>2}/{:>2} | pp={:>5}\n'.format(epoch, srn.num_epochs, int(pp)))
    #
    plot_ba_trajs(num_cats2bas, num_cats2max_ba,
                  title='NUM_TOKENS={} MAX_NGRAM_SIZE={} NUM_DESCENDANTS={} NUM_LEVELS={} E={} ZIPF_A={}'.format(
                      NUM_TOKENS, MAX_NGRAM_SIZE, NUM_DESCENDANTS, NUM_LEVELS, E, ZIPF_A))