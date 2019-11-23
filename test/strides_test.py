import unittest
import numpy as np

from preppy.legacy import TrainPrep

from startingsmall import config
from startingsmall.docs import load_docs
from startingsmall.params import param2default


class MyTest(unittest.TestCase):

    @staticmethod
    def test_gen_windows():
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
                               )

        a1 = [w for w in train_prep.gen_windows(reordered_parts=list(train_prep.reordered_halves))]
        a2 = [w for w in train_prep.gen_windows(reordered_parts=list(train_prep.reordered_parts))]

        return np.array_equal(a1, a2)


if __name__ == '__main__':
    unittest.main()