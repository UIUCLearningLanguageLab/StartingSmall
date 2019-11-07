import random
from typing import List,Tuple, Optional
from pathlib import Path


def load_docs(corpus_path: Path,
              shuffle_docs: bool,
              test_doc_ids: Optional[List[int]] = None,
              num_test_docs: Optional[int] = 100,
              shuffle_seed: Optional[int] = 20,
              split_seed: Optional[int] = 3
              ) -> Tuple[List[str], List[str]]:
    """
    100 test docs and split_seed = 3 were used in PH master's thesis
    """

    # load CHILDES transcripts as list of strings
    docs = corpus_path.read_text().split('\n')
    num_docs = len(docs)
    print(f'Loaded {num_docs} documents from {corpus_path}')

    # split train/test
    print('Splitting docs into train and test...')
    if test_doc_ids is None:
        num_test_doc_ids = num_docs - num_test_docs
        random.seed(split_seed)
        test_doc_ids = random.sample(range(num_test_doc_ids), num_test_docs)
    else:
        print('WARNING: Using custom test-doc-ids.')

    test_docs = []
    for test_line_id in test_doc_ids:
        test_doc = docs.pop(test_line_id)  # removes line and returns removed line
        test_docs.append(test_doc)

    # shuffle after train/test split
    if shuffle_docs:
        print('Shuffling documents')
        random.seed(shuffle_seed)
        random.shuffle(docs)

    print(f'Collected {len(docs):,} train docs')
    print(f'Collected {len(test_docs):,} test docs')

    return docs, test_docs
