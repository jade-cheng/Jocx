import unittest

from ziphmm import Matrix
from _internal import seq_to_string
from _internal import format_matrix


class TestMatrix(unittest.TestCase):

    def test_compute_sum(self):
        m = Matrix(2, 3)
        m[0, 0] = 1
        m[0, 1] = 2
        m[0, 2] = 3
        m[1, 0] = 4
        m[1, 1] = 5
        m[1, 2] = 6
        self.assertEqual(21, m.compute_sum())

    def test_indexers(self):
        m = Matrix(2, 3)
        m[0, 0] = 1
        m[0, 1] = 2
        m[0, 2] = 3
        m[1, 0] = 4
        m[1, 1] = 5
        m[1, 2] = 6

        self.assertEqual(1, m[0, 0])
        self.assertEqual(2, m[0, 1])
        self.assertEqual(3, m[0, 2])
        self.assertEqual(4, m[1, 0])
        self.assertEqual(5, m[1, 1])
        self.assertEqual(6, m[1, 2])

    def test_init(self):
        m = Matrix(2, 3)
        self.assertEqual(2, m.height)
        self.assertEqual(3, m.width)
        self.assertEqual('{{0.0,0.0,0.0},{0.0,0.0,0.0}}', format_matrix(m))

    def test_mul(self):
        m = Matrix(2, 3)
        m[0, 0] = 1
        m[0, 1] = 2
        m[0, 2] = 3
        m[1, 0] = 4
        m[1, 1] = 5
        m[1, 2] = 6

        n = Matrix(3, 2)
        n[0, 0] = 1
        n[0, 1] = 2
        n[1, 0] = 3
        n[1, 1] = 4
        n[2, 0] = 5
        n[2, 1] = 6

        q = m * n
        self.assertEqual('{{22.0,28.0},{49.0,64.0}}', format_matrix(q))

    def test_scale(self):
        m = Matrix(2, 3)
        m[0, 0] = 1
        m[0, 1] = 2
        m[0, 2] = 3
        m[1, 0] = 4
        m[1, 1] = 5
        m[1, 2] = 6
        m.scale(2)
        self.assertEqual('{{2.0,4.0,6.0},{8.0,10.0,12.0}}', format_matrix(m))

if __name__ == '__main__':
    unittest.main()
