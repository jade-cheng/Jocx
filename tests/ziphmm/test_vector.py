import unittest

from ziphmm import Vector
from _internal import format_vector


class TestVector(unittest.TestCase):

    def test_indexers(self):
        v = Vector(3)
        v[0] = 1
        v[1] = 2
        v[2] = 3

        self.assertEqual(1, v[0])
        self.assertEqual(2, v[1])
        self.assertEqual(3, v[2])

    def test_init(self):
        v = Vector(3)
        self.assertEqual('{0.0,0.0,0.0}', format_vector(v))

if __name__ == '__main__':
    unittest.main()
