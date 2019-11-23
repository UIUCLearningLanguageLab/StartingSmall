import unittest
import numpy as np

from preppy.legacy import TrainPrep

from startingsmall import config
from startingsmall.docs import load_docs
from startingsmall.params import param2default


class MyTest(unittest.TestCase):

    def test_gen_windows(self):
        """
        test that Preppy legacy.TrainPrep.gen_windows() works as expected
        """

        corpus_path = config.Dirs.data / f'{param2default["corpus"]}.txt'
        train_docs, test_docs = load_docs(corpus_path,
                                          param2default['shuffle_docs'])

        num_parts = 2
        num_iterations = [1, 2]
        reverse = True
        train_prep = TrainPrep(train_docs,
                               reverse,
                               param2default['num_types'],
                               num_parts,
                               num_iterations,
                               param2default['batch_size'],
                               param2default['context_size'],
                               config.Eval.num_evaluations,
                               shuffle_within_part=False,
                               )

        gold_parts = [train_prep.store.token_ids[:train_prep.midpoint],
                      train_prep.store.token_ids[-train_prep.midpoint:]]
        if reverse:
            gold_parts = gold_parts[::-1]

        a1 = [w for w in train_prep.gen_windows(reordered_parts=list(gold_parts))]
        a2 = [w for w in train_prep.gen_windows(reordered_parts=list(train_prep.reordered_parts))]

        self.assertTrue(np.array_equal(a1, a2))


if __name__ == '__main__':
    unittest.main()