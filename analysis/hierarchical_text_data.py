import numpy as np
import pyprind


def generate_tokens_from_zipfian(vocab, num_tokens):  # TODO use
    num_vocab = len(vocab)
    res = [vocab[i] if i < num_vocab else 'OOV' for i in np.random.zipf(2, num_tokens)]
    return res


def sample_using_hierarchical_diffusion(num_descendants, num_levels, e):
    """the higher the change probability (e),
     the less variance accounted for by more distant nodes in diffusion process"""
    node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
    nodes = [node0]
    for level in range(num_levels):
        candidate_nodes = nodes * num_descendants
        nodes = [node if p else -node for node, p in zip(candidate_nodes,
                                                         np.random.binomial(n=2, p=1-e, size=len(candidate_nodes)))]
    return nodes


def make_data(num_tokens, max_ngram_size=6, num_descendants=2, num_levels=12, e=0.01, verbose=False):
    """
    generate text by adding one word at a time to a list of words.
    each word is constrained by the structure matrices - which are hierarchical -
    and determine the legal successors for each word in teh vocabulary.
    there is one structure matrix for each ngram (in other words, for each distance up to MAX_NGRAM_SIZE)
    the set of legal words that can follow a word is the intersection of legal words dictated by each structure matrix
    the size of the intersection is approximately the same for each word, and a word is sampled uniformly from this set
    the size of the intersection is the best possible perplexity a model can achieve,
    because perplexity indicates the number of choices from a random uniformly distributed set of choices

    """
    # vocab
    num_vocab = num_descendants ** num_levels
    print('num_vocab={}'.format(num_vocab))
    vocab = ['w{}'.format(i) for i in np.arange(num_vocab)]
    word2id = {word: n for n, word in enumerate(vocab)}
    # check
    num_theoretical_legals = num_vocab / (2 ** max_ngram_size)
    print('num_theoretical_legals={}'.format(num_theoretical_legals))  # perplexity should converge to this value
    # ngram2structure_mat - col words are features that predict row words
    ngram_sizes = range(1, max_ngram_size + 1)
    ngram2structure_mat = {ngram: np.zeros((num_vocab, num_vocab), dtype=np.int) for ngram in ngram_sizes}
    for ngram_size in ngram_sizes:
        print('Making structure_mat with ngram_size={}...'.format(ngram_size))
        for n in range(num_vocab):
            ngram2structure_mat[ngram_size][n] = sample_using_hierarchical_diffusion(num_descendants, num_levels, e)
    # tokens
    num_no_legals = 0
    tokens = np.random.choice(vocab, size=1).tolist()
    pbar = pyprind.ProgBar(num_tokens)
    for _ in range(num_tokens):
        legal_nexts = set(vocab)
        for size in ngram_sizes:
            try:
                previous_token = tokens[-size]
            except IndexError:
                print('WARNING: Did not find any token {} steps back'.format(size))
                break
            else:
                structure_mat = ngram2structure_mat[size]
                col = structure_mat[:, word2id[previous_token]]
                legal_nexts.intersection_update([w for n, w in enumerate(vocab) if col[n] == 1])
            if verbose:
                print(size, len(legal_nexts))
        #
        if not legal_nexts:
            num_no_legals += 1
            new_token = np.random.choice([w for w in vocab if w not in tokens[-max_ngram_size:]],
                                         size=1).item()
        else:
            new_token = np.random.choice(list(legal_nexts), size=1).item()
        tokens.append(new_token)
        pbar.update()
    if verbose:
        print(tokens)
        print('num_no_legals={}/{}'.format(num_no_legals, num_tokens))  # important to check this
    token_ids = [word2id[w] for w in tokens]
    return tokens, token_ids