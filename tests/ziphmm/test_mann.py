import unittest

from ziphmm import _mann
from _internal import create_hmm
from _internal import create_seq


class TestMann(unittest.TestCase):

    def test0(self):
        self.__test('test0', 'test0', -12.4766)

    def test1(self):
        self.__test('test1', 'test1', -12.5671)

    def test2(self):
        self.__test('test2', 'test2', -27.5310)

    def test3(self):
        self.__test('test3', 'test3', 0.0)

    def test4(self):
        self.__test('test4', 'test4', -24.0953)

    def test4_10000(self):
        self.__test('test4', 'test4_10000', -44239.7313)

    def test5(self):
        self.__test('test5', 'test5', -5.7827)

    def test6(self):
        self.__test('test6', 'test6', -18.8829)

    def test6_10000(self):
        self.__test('test6', 'test6_10000', -10457.5211)

    def test6_100000(self):
        self.__test('test6', 'test6_100000', -104126.2411)

    def test_8_states(self):
        self.__test('test8States', 'test8States', 0.0)

    def test_off_by_one_em(self):
        self.__test('testOffByOneEm', 'testOffByOneEm', -16.6355)

    def test_off_by_one_trans(self):
        self.__test('TestOffByOneTrans', 'TestOffByOneTrans', -0.0000)

    def test_one_state(self):
        self.__test('testOneState', 'testOneState', -12.4766)

    def test_zero_add(self):
        self.__test('testZeroAdd', 'testZeroAdd', None)

    def test_no_solution(self):
        self.__test('testNoSolution', 'testNoSolution', None)

    def __test(self, hmm_name, seq_name, expected_value):
        def fix(n, digits=4):
            if n is None:
                return None
            fmt = '{0:.' + str(digits) + 'f}'
            return fmt.format(round(n, digits))

        hmm = create_hmm('inputs/' + hmm_name + '.hmm')
        seq = create_seq('inputs/' + seq_name + '.seq')
        actual_value = _mann.evaluate(hmm, seq)
        self.assertEqual(fix(expected_value), fix(actual_value))

if __name__ == '__main__':
    unittest.main()
