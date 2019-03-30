import numpy as np

MAX_NGRAM_SIZE = 3
NUM_TOKENS = 1000

E = 0.1  # the higher, the less variance accounted for by more distant nodes in diffusion process
NUM_DESCENDANTS = 2  # 2
NUM_LEVELS = 5  # 9


def generate_tokens_from_zipfian(vocab, num_tokens):  # TODO use
    num_vocab = len(vocab)
    res = [vocab[i] if i < num_vocab else 'OOV' for i in np.random.zipf(2, num_tokens)]
    return res


def sample_using_hierarchical_diffusion(num_levels=NUM_LEVELS, num_descendants=NUM_DESCENDANTS, e=E):
    node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
    nodes = [node0]
    for level in range(num_levels):
        candidate_nodes = nodes * num_descendants
        nodes = [node if np.random.binomial(n=1, p=1-e) else -node for node in candidate_nodes]
    return nodes


def get_new_token(previous_tokens):
    legals_list = []
    for size in ngram_sizes:
        try:
            previous_token = previous_tokens[-size]
        except IndexError:
            break
        else:
            structure_mat = ngram2structure_mat[size]
            row = structure_mat[word2id[previous_token]]
            legals = [w for n, w in enumerate(vocab) if row[n] == 1]
            legals_list.append(legals)
    legal_nexts = list(set(legals_list[0]).intersection(*legals_list))
    if not legal_nexts:
        print('WARNING: No legal next word for {}'.format(previous_tokens[-MAX_NGRAM_SIZE:]))
        return np.random.choice(vocab, size=1).item()  # TODO good idea?
    else:
        return np.random.choice(legal_nexts, size=1).item()


# vocab
num_vocab = NUM_DESCENDANTS ** NUM_LEVELS
vocab = ['word{}'.format(i) for i in np.arange(num_vocab)]
word2id = {word: n for n, word in enumerate(vocab)}

# ngram2structure_mat
ngram_sizes = range(1, MAX_NGRAM_SIZE + 1)
ngram2structure_mat = {ngram: np.zeros((num_vocab, num_vocab), dtype=np.int) for ngram in ngram_sizes}
for ngram_size in ngram_sizes:
    for n in range(num_vocab):
        ngram2structure_mat[ngram_size][n] = sample_using_hierarchical_diffusion()

# tokens
tokens = np.random.choice(vocab, size=1).tolist()
for _ in range(NUM_TOKENS):
    new_token = get_new_token(tokens)
    tokens.append(new_token)


print(tokens)