import random

from startingsmall import config


def load_docs(params, num_test_docs=100, seed=3):
    """
    100 test docs + random seed = 3 were used in PH master's thesis
    """
    # load input as a list of strings (CHILDES transcripts or Wiki articles)
    with (config.RemoteDirs.data / f'{params.corpus}.txt').open('r') as f:
        docs = f.readlines()
    num_docs = len(docs)
    print(f'Loaded {num_docs} documents from {params.corpus}')

    if params.shuffle_docs:
        random.shuffle(docs)

    # split train/test
    num_test_doc_ids = num_docs - num_test_docs
    random.seed(seed)
    test_doc_ids = random.sample(range(num_test_doc_ids), num_test_docs)
    test_docs = []
    for test_line_id in test_doc_ids:
        test_doc = docs.pop(test_line_id)  # removes line and returns removed line
        test_docs.append(test_doc)

    return docs, test_docs