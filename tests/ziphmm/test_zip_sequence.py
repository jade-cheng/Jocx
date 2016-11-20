import unittest

from ziphmm import ZipSequence
from _internal import create_hmm
from _internal import create_seq
from _internal import seq_to_string


class TestZipSequence(unittest.TestCase):

    def test_from_sequence(self):
        hmm = create_hmm("inputs/test0.hmm")
        seq = create_seq("inputs/test0.seq")
        xseq = ZipSequence.from_sequence(seq, hmm.state_count)

        self.assertEqual('0 1 4 4', seq_to_string(xseq))

        self.assertEqual(5, xseq.alphabet_size)
        self.assertEqual(3, xseq.extended_alphabet_size)
        self.assertEqual(2, xseq.original_alphabet_size)
        self.assertEqual(2, xseq.state_count)

        self.assertEqual((3, 3), xseq.get_substitute(4))
        self.assertEqual((2, 2), xseq.get_substitute(3))
        self.assertEqual((0, 1), xseq.get_substitute(2))

        self.assertIsNone(xseq.get_substitute(1))

        self.assertEqual(
            '0 1 2 3 4',
            seq_to_string(xseq.iterate_alphabet()))

        self.assertEqual(
            '2 3 4',
            seq_to_string(xseq.iterate_extended_alphabet()))

        self.assertEqual(
            '0 1',
            seq_to_string(xseq.iterate_original_alphabet()))

        self.assertEqual(
            '0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1',
            seq_to_string(xseq.iterate_original_sequence()))

        self.assertEqual(
            '2 0 1 ; 3 2 2 ; 4 3 3',
            ' ; '.join(map(lambda (a, (b, c)): ' '.join(
                map(str, [a, b, c])), xseq.iterate_substitutes())))

    def test0(self):
        self.__test('test0', 'test0')

    def test1(self):
        self.__test('test1', 'test1')

    def test2(self):
        self.__test('test2', 'test2')

    def test3(self):
        self.__test('test3', 'test3')

    def test4(self):
        self.__test('test4', 'test4')

    def test_5(self):
        self.__test('test5', 'test5')

    def test_5_20(self):
        self.__test('test5', 'test5_20')

    def test_6(self):
        self.__test('test6', 'test6')

    def test_8_states(self):
        self.__test('test8States', 'test8States')

    def __test(self, hmm, seq):
        hmm = create_hmm('inputs/' + hmm + '.hmm')
        seq = create_seq('inputs/' + seq + '.seq')
        xseq = ZipSequence.from_sequence(seq, hmm.state_count)
        expect = seq_to_string(seq)
        actual = seq_to_string(xseq.iterate_original_sequence())
        self.assertEqual(expect, actual)

    #
    # These also work, but compressing the sequences takes time so ignore these
    # test sequences.
    #
    # def test4_10000(self):
    #     self.__test('test4', 'test4_10000')
    #
    # def test_6_10000(self):
    #     self.__test('test6', 'test6_10000')
    #
    # def test_6_100000(self):
    #     self.__test('test6', 'test6_100000')
    #


if __name__ == '__main__':
    unittest.main()
