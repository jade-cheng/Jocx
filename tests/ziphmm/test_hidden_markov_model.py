import os
import unittest

from ziphmm import HiddenMarkovModel
from _internal import TempDirectory
from _internal import create_hmm
from _internal import format_matrix
from _internal import format_vector
from _internal import locate


class TestHiddenMarkovModel(unittest.TestCase):

    def test_init(self):
        hmm = HiddenMarkovModel(2, 3)
        self.assertEqual(3, hmm.alphabet_size)
        self.assertEqual(2, hmm.state_count)

        a_str = '{{0.0,0.0},{0.0,0.0}}'
        self.assertEqual(a_str, format_matrix(hmm.a))

        b_str = '{{0.0,0.0,0.0},{0.0,0.0,0.0}}'
        self.assertEqual(b_str, format_matrix(hmm.b))

        pi_str = '{0.0,0.0}'
        self.assertEqual(pi_str, format_vector(hmm.pi))

    def test_from_file(self):
        path = locate('inputs/test2.hmm')
        hmm = HiddenMarkovModel.from_file(path)
        self.assertEqual(4, hmm.alphabet_size)
        self.assertEqual(3, hmm.state_count)

        a_str = '{{0.1,0.2,0.7},{0.3,0.4,0.3},{0.5,0.5,0.0}}'
        self.assertEqual(a_str, format_matrix(hmm.a))

        b_str = '{{0.1,0.2,0.3,0.4},{0.2,0.3,0.4,0.1},{0.3,0.4,0.1,0.2}}'
        self.assertEqual(b_str, format_matrix(hmm.b))

        pi_str = '{0.1,0.2,0.7}'
        self.assertEqual(pi_str, format_vector(hmm.pi))

    def test_save(self):
        hmm = create_hmm('inputs/test2.hmm')
        with TempDirectory() as temp:
            hmm_path = os.path.join(temp.path, 'file.hmm')
            hmm.save(hmm_path)

            with open(hmm_path, 'r') as factual:
                actual = factual.read().strip()

                with open(locate('inputs/test2.hmm')) as fexpect:
                    expect = fexpect.read().strip()

                    self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
