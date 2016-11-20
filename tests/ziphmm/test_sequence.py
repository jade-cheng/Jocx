import unittest

from ziphmm import Sequence
from _internal import create_seq
from _internal import seq_to_string


class TestSequence(unittest.TestCase):

    def test_init(self):
        seq = Sequence((1, 2, 3), 5)
        self.assertEqual(3, len(seq))
        self.assertEqual(5, seq.alphabet_size)
        self.assertEqual(1, seq[0])
        self.assertEqual(2, seq[1])
        self.assertEqual(3, seq[2])

        self.assertEqual(
            '0 1 2 3 4',
            seq_to_string(seq.iterate_alphabet()))

    def test_from_file(self):
        seq = create_seq('inputs/test2.seq')
        self.assertEqual(20, len(seq))
        self.assertEqual(4, seq.alphabet_size)

        self.assertEqual(
            '0 2 3 1 0 2 0 2 2 1 0 2 3 0 2 1 0 3 2 1',
            seq_to_string(seq))

if __name__ == '__main__':
    unittest.main()
