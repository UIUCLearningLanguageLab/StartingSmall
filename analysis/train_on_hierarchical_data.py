from cytoolz import itertoolz

from analysis.hierarchical_text_data import make_data
from analysis.rnn import RNN


NUM_TOKENS = 1000 * 10  # a small number of tokens can result in less then theoretically predicted perplexity
MAX_NGRAM_SIZE = 3  # TODO
NUM_DESCENDANTS = 2
NUM_LEVELS = 12
E = 0.01

MB_SIZE = 64

# train_seqs
tokens, token_ids = make_data(NUM_TOKENS, MAX_NGRAM_SIZE, NUM_DESCENDANTS, NUM_LEVELS, E, verbose=True)
train_seqs = []
for seq in itertoolz.partition_all(MB_SIZE, token_ids):  # a seq contains 64 token_ids
    if len(seq) == MB_SIZE:
        train_seqs.append(list(seq))  # need to convert tuple to list
    else:
        print('WARNING: Found sequence of length != {}'.format(MB_SIZE))
print('num sequences={}'.format(len(train_seqs)))

# srn
input_size = NUM_DESCENDANTS ** NUM_LEVELS
srn = RNN(input_size, learning_rate=(0.01, 0.00, 20), num_seqs_in_batch=1)  # num_seqs_in_batch must be 1

# feed a list of sequences (each seq is a list of term_ids)
srn.train(train_seqs, verbose=False)