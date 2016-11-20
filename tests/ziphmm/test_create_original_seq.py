import os
import random
import time
import unittest

from _internal import TempDirectory
from ziphmm import ZipDirectory


class TestCreateOriginalSequence(unittest.TestCase):

    def test_speed(self):
        _LINE_LENGTH = 100

        sym_count = 1
        while sym_count < 100000:
            sym_count *= 10

            with TempDirectory() as tmp:

                t1 = time.clock()

                for p in ('a.fa', 'b.fa'):
                    with open(os.path.join(tmp.path, p), 'w', sym_count) as f:
                        f.write('> seq\n')
                        for j in xrange(sym_count):
                            f.write(random.choice(['a', 'c', 't', 'g', '-']))
                            if (j + 1) % _LINE_LENGTH == 0:
                                f.write('\n')
                        f.write('\n')

                t2 = time.clock()

                ZipDirectory.create_original_sequences(
                    tmp.path,
                    os.path.join(tmp.path, 'a.fa'),
                    os.path.join(tmp.path, 'b.fa'))

                t3 = time.clock()
                self.assertTrue(t1 < t2 < t3)
