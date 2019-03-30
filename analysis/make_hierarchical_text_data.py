import numpy as np

VERBOSE = False

MAX_NGRAM_SIZE = 6
NUM_TOKENS = 100

E = 0.01  # the higher, the less variance accounted for by more distant nodes in diffusion process
NUM_DESCENDANTS = 2  # 2
NUM_LEVELS = 12  # 12


num_no_legals = 0


def generate_tokens_from_zipfian(vocab, num_tokens):  # TODO use
    num_vocab = len(vocab)
    res = [vocab[i] if i < num_vocab else 'OOV' for i in np.random.zipf(2, num_tokens)]
    return res


def sample_using_hierarchical_diffusion(num_levels=NUM_LEVELS, num_descendants=NUM_DESCENDANTS, e=E):
    node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
    nodes = [node0]
    for level in range(num_levels):
        candidate_nodes = nodes * num_descendants
        nodes = [node if p else -node for node, p in zip(candidate_nodes,
                                                         np.random.binomial(n=2, p=1-e, size=len(candidate_nodes)))]
    return nodes


def get_new_token(previous_tokens):
    global num_no_legals
    legal_nexts = set(vocab)
    for size in ngram_sizes:
        try:
            previous_token = previous_tokens[-size]
        except IndexError:
            print('WARNING: Did not find any token {} steps back'.format(size))
            break
        else:
            structure_mat = ngram2structure_mat[size]
            col = structure_mat[:, word2id[previous_token]]
            legal_nexts.intersection_update([w for n, w in enumerate(vocab) if col[n] == 1])
        print(size, len(legal_nexts))
    #
    print('num legals={}'.format(len(legal_nexts)))
    if not legal_nexts:
        num_no_legals += 1
        return np.random.choice([w for w in vocab if w not in previous_tokens[-MAX_NGRAM_SIZE:]],
                                size=1).item()
    else:
        return np.random.choice(list(legal_nexts), size=1).item()


# vocab
num_vocab = NUM_DESCENDANTS ** NUM_LEVELS
print('num_vocab={}'.format(num_vocab))
vocab = ['w{}'.format(i) for i in np.arange(num_vocab)]
word2id = {word: n for n, word in enumerate(vocab)}

# check
num_theoretical_legals = num_vocab / (2 ** MAX_NGRAM_SIZE)
print('num_theoretical_legals={}'.format(num_theoretical_legals))  # perplexity should converge to this value


# ngram2structure_mat - col words are features that predict row words
ngram_sizes = range(1, MAX_NGRAM_SIZE + 1)
ngram2structure_mat = {ngram: np.zeros((num_vocab, num_vocab), dtype=np.int) for ngram in ngram_sizes}
for ngram_size in ngram_sizes:
    print('Making structure_mat with ngram_size={}...'.format(ngram_size))
    for n in range(num_vocab):
        ngram2structure_mat[ngram_size][n] = sample_using_hierarchical_diffusion()
    if VERBOSE:
        for col in ngram2structure_mat[ngram_size].T:
            print(len(np.where(col == 1)[0]))

# tokens
tokens = np.random.choice(vocab, size=1).tolist()
for _ in range(NUM_TOKENS):
    new_token = get_new_token(tokens)
    tokens.append(new_token)


print(tokens)
print('num_no_legals={}/{}'.format(num_no_legals, NUM_TOKENS)) # important to check this