import os
import unittest

from ziphmm import ZipDirectory
from _internal import TempDirectory
from _internal import create_ziphmm_directory
from _internal import locate


class TestZipDirectory(unittest.TestCase):

    def test_clear_cache(self):
        with TempDirectory() as temp:
            directory = create_ziphmm_directory(temp.path)
            xdir = ZipDirectory(directory)
            self.assertTrue(os.path.isfile(
                os.path.join(directory, 'data_structure')))
            xdir.clear_cache()
            self.assertFalse(os.path.isfile(
                os.path.join(directory, 'data_structure')))

    def test_create_cache(self):
        with TempDirectory() as temp:
            directory = create_ziphmm_directory(temp.path)
            xdir = ZipDirectory(directory)
            self.assertFalse(xdir.is_cached(3))
            xdir.create_cache(3)
            self.assertTrue(xdir.is_cached(3))

    def test_create_original_sequences(self):
        with TempDirectory() as temp:
            ZipDirectory.create_original_sequences(
                temp.path,
                locate("inputs/a.fa"),
                locate("inputs/b.fa"),
                chunk_size=10)

            def test(name, expected):
                path = os.path.join(temp.path, name, 'original_sequence')
                self.assertTrue(os.path.isfile(path))
                actual = open(path, 'r').read()
                self.assertEqual(expected, actual)

            test('s1.ziphmm0', '0 0 0 0 1 1 1 1 0 0 ')
            test('s1.ziphmm1', '0 0 1 1 1 1 0 0 0 0 ')
            test('s1.ziphmm2', '1 1 2 1 ')
            test('s2.ziphmm0', '0 0 0 0 1 1 0 0 0 0 ')
            test('s2.ziphmm1', '1 1 0 0 0 0 1 1 ')

    def test_has_original_sequence(self):
        with TempDirectory() as temp:
            self.assertFalse(ZipDirectory.has_original_sequence(temp.path))
            directory = create_ziphmm_directory(temp.path)
            self.assertTrue(ZipDirectory.has_original_sequence(directory))

    def test_init_with_data(self):
        with TempDirectory() as temp:
            directory = create_ziphmm_directory(temp.path)
            xdir = ZipDirectory(directory)
            self.assertEqual(directory, xdir.path)

    def test_init_without_data(self):
        with TempDirectory() as temp:
            directory = create_ziphmm_directory(temp.path)
            data_structure_path = os.path.join(directory, 'data_structure')
            os.remove(data_structure_path)
            xdir = ZipDirectory(directory)
            self.assertEqual(directory, xdir.path)

    def test_is_cached(self):
        with TempDirectory() as temp:
            directory = create_ziphmm_directory(temp.path)
            xdir = ZipDirectory(directory)
            self.assertTrue(xdir.is_cached(2))
            self.assertFalse(xdir.is_cached(3))
            xdir.create_cache(3)
            self.assertTrue(xdir.is_cached(2))
            self.assertTrue(xdir.is_cached(3))
            xdir.clear_cache()
            self.assertFalse(xdir.is_cached(2))
            self.assertFalse(xdir.is_cached(3))

    def test_load(self):
        with TempDirectory() as temp:
            directory = create_ziphmm_directory(temp.path)
            xdir = ZipDirectory(directory)

            self.assertTrue(xdir.is_cached(2))
            xseq2 = xdir.load(2)
            self.assertEqual(2, xseq2.state_count)
            self.assertTrue(xdir.is_cached(2))

            self.assertFalse(xdir.is_cached(3))
            xseq3 = xdir.load(3)
            self.assertEqual(3, xseq3.state_count)
            self.assertTrue(xdir.is_cached(3))

            self.assertFalse(xdir.is_cached(4))
            xseq4 = xdir.load(4)
            self.assertEqual(4, xseq4.state_count)
            self.assertTrue(xdir.is_cached(4))

    def test_path(self):
        with TempDirectory() as temp:
            directory = create_ziphmm_directory(temp.path)
            xdir = ZipDirectory(directory)
            self.assertEqual(directory, xdir.path)

if __name__ == '__main__':
    unittest.main()
