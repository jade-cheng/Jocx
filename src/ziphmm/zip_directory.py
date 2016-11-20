import os
import shutil

from sequence import Sequence
from zip_sequence import ZipSequence
from _create_original_seq import create_original_sequence
from _internal import read_symbol_array
from _token_scanner import TokenScanner

#
# NOTE This class assumes the sequences stored in the ZipHMM directories have
# not changed since the caches were created and the 'original_sequence' and
# 'data_structure' files have not been corrupted.
#


def _original_sequence_path(ziphmm_dir):
        return os.path.join(ziphmm_dir, 'original_sequence')


class ZipDirectory:
    """
    Implementation of an object that manages a cache of zipped sequences.  This
    class assumes the sequences stored in the ZipHMM directories have not
    changed since the caches were created and that the 'original sequence' and
    associated 'data structure' have not been corrupted.
    """

    def __init__(self, directory):
        """
        Initialize a new instance of the ZipDirectory class based on the
        specified directory.  Require that the directory contains at least the
        text file, 'original_sequence', which contains the symbols of the
        original sequence in order and separated by whitespace.

        directory -- The directory to manage.
        """

        assert os.path.isdir(directory)

        #
        # All ZipHMM directories must contain the 'original_sequence' file.
        #
        self.__path = directory
        assert os.path.isfile(self.__original_sequence_path)

        #
        # If the 'data_structure' file does not exist, this is an uninitialized
        # directory; read the original sequence and initially store the data
        # structure.
        #
        if not os.path.isfile(self.__data_structure_path):
            seq = Sequence.from_file(self.__original_sequence_path)
            self.__unzipped_alphabet_size = seq.alphabet_size
            self.__unzipped_sequence_length = len(seq)
            self.__zipped_alphabet_sizes = dict()
            self.__substitutions = dict()
            self.__save_data_structure()
            return

        #
        # Read the original alphabet size; only 16-bit values are allowed, and
        # there must be at least one symbol in the alphabet.
        #
        scanner = TokenScanner(self.__data_structure_path)
        scanner.require('orig_alphabet_size')
        self.__unzipped_alphabet_size = scanner.read_int()
        assert 1 <= self.__unzipped_alphabet_size < 65536

        #
        # Read the original sequence length and assume this still matches the
        # data in the original sequence itself.
        #
        scanner.require('orig_seq_length')
        self.__unzipped_sequence_length = scanner.read_int()
        assert 1 <= self.__unzipped_sequence_length

        #
        # Read the number of symbols compressed for various state counts; this
        # will also define the maximum symbol value.
        #
        scanner.require('nStates2alphabet_size')
        self.__zipped_alphabet_sizes = dict()
        max_alphabet_size = 0
        while True:
            token = scanner.peek()
            if token is None or token == 'symbol2pair':
                break

            #
            # Read the state count; duplicate entries are not allowed.
            #
            state_count = scanner.read_int()
            assert state_count > 0
            assert state_count not in self.__zipped_alphabet_sizes

            #
            # Read the alphabet size.
            #
            alphabet_size = scanner.read_int()
            assert alphabet_size > 0
            max_alphabet_size = max(alphabet_size, max_alphabet_size)

            #
            # Store the state count and alphabet size into the table used to
            # decompressed zipped sequences.
            #
            self.__zipped_alphabet_sizes[state_count] = alphabet_size

        #
        # Read the substitution table.
        #
        self.__substitutions = dict()
        scanner.require('symbol2pair')
        while scanner.peek() is not None:
            #
            # Read the new symbol; disallow duplicate entries for new symbols,
            # ensure the new symbol is not in the original alphabet, and ensure
            # the new symbol is not larger than the largest recorded alphabet
            # size from the previous section.
            #
            new_symbol = scanner.read_int()
            assert new_symbol not in self.__substitutions
            assert new_symbol >= self.__unzipped_alphabet_size
            assert new_symbol < max_alphabet_size

            #
            # Read the two symbols; neither may be larger than the largest
            # recorded alphabet size.
            #
            a = scanner.read_int()
            b = scanner.read_int()
            assert a < max_alphabet_size
            assert b < max_alphabet_size

            #
            # Record the entry in the substitution table, new_symbol => (a, b).
            #
            self.__substitutions[new_symbol] = (a, b)

        #
        # All (a, b) values symbols in the substitution table must either be
        # from the original alphabet or be present as a new symbol in the
        # substitution table.
        # subst
        #
        assert all(map(lambda (key, (a_, b_)):
                       (a_ < self.__unzipped_alphabet_size or
                        a_ in self.__substitutions) and
                       (b_ < self.__unzipped_alphabet_size or
                        b_ in self.__substitutions),
                       self.__substitutions.iteritems()))

    def __repr__(self):
        return self.path

    @property
    def path(self):
        """ Return the path to the directory managed by this instance. """
        return self.__path

    @property
    def __data_structure_path(self):
        return os.path.join(self.__path, 'data_structure')

    @property
    def __original_sequence_path(self):
        return _original_sequence_path(self.__path)

    @property
    def __state_count_to_sequence_path(self):
        return os.path.join(self.__path, 'nStates2seq')

    def clear_cache(self):
        """
        Delete all files in the directory other than the original sequence.
        """
        if os.path.isdir(self.__state_count_to_sequence_path):
            shutil.rmtree(self.__state_count_to_sequence_path)

        if os.path.isfile(self.__data_structure_path):
            os.remove(self.__data_structure_path)

        self.__zipped_alphabet_sizes = dict()
        self.__substitutions = dict()

    def create_cache(self, state_count):
        """
        Create a new zipped sequence optimized for the specified number of
        states in the hidden Markov model and cache the new sequence in the
        directory managed by this instance.

        state_count -- The optimal number of states in the hidden Markov model.
        """

        assert state_count > 0

        #
        # Create the directory of zipped sequences if it does not yet exist.
        #
        if not os.path.isdir(self.__state_count_to_sequence_path):
            os.mkdir(self.__state_count_to_sequence_path)

        #
        # Reread the original sequence, and zip it, optimized for the specified
        # state count.
        #
        seq = Sequence.from_file(
            self.__original_sequence_path,
            self.__unzipped_alphabet_size)

        x_seq = ZipSequence.from_sequence(seq, state_count)

        #
        # Loop over symbols in the zipped sequence; if the symbol is new, then
        # ensure it matches an existing symbol in the substitution table--or
        # that it does not exist in the substitution table; put new symbols
        # into the substitution table of this instance.
        #
        for symbol in xrange(x_seq.alphabet_size):
            if symbol < self.__unzipped_alphabet_size:
                continue

            new_a, new_b = x_seq.get_substitute(symbol)
            if symbol in self.__substitutions:
                old_a, old_b = self.__substitutions[symbol]
                assert old_a == new_a
                assert old_b == new_b
                continue

            self.__substitutions[symbol] = (new_a, new_b)

        #
        # Record the number of symbols used for this state count, save the new
        # data structure, save the zipped sequence, and return the sequence.
        #
        self.__zipped_alphabet_sizes[state_count] = x_seq.alphabet_size
        self.__save_data_structure()
        x_seq_path = self.__resolve_zipped_sequence_path(state_count)
        x_seq.save(x_seq_path)
        return x_seq

    @staticmethod
    def create_original_sequences(ziphmm_dir, fasta1, fasta2, chunk_size=None, logger=None):
        """
        Create ZipHMM original sequences for two FASTA files, if the sequences
        do not already exist. When creating these cached files, write output
        to the specified logger, if one is specified. If a chunk size is
        specified, split the sequences from the FASTA files into multiple
        ZipHMM directories of equal or lesser size. If an error occurs raise
        an IOError or RuntimeError.

        For example, if...

          * ziphmm_dir is '/opt/data'
          * the two fasta files have sequences named 's1', 's2', and 's3'
          * the sequences are all of length 1000
          * the chunk size is specified as 500

        ...then create the following files:

          /opt/data/s1.ziphmm0/original_sequence
          /opt/data/s1.ziphmm1/original_sequence
          /opt/data/s2.ziphmm0/original_sequence
          /opt/data/s2.ziphmm1/original_sequence
          /opt/data/s3.ziphmm0/original_sequence
          /opt/data/s3.ziphmm1/original_sequence

        If no chunk size is specified, create the following files:

          /opt/data/s1.ziphmm/original_sequence
          /opt/data/s2.ziphmm/original_sequence
          /opt/data/s3.ziphmm/original_sequence

        If the FASTA files have different sequence names, or if the length of
        the sequences differ, write a warning message to stderr, ignore the
        offending sequences, and continue processing the files without raise
        an IOError or RuntimeError.

        ziphmm_dir -- The root of the ZipHMM directories.
        fasta1 --     The path to the first FASTA file.
        fasta2 --     The path to the second FASTA file.
        chunk_size -- The chunk size, or None.
        logger --     The logger, or None.
        """

        create_original_sequence(
            ziphmm_dir,
            fasta1,
            fasta2,
            chunk_size,
            logger)

    @staticmethod
    def has_original_sequence(ziphmm_dir):
        original_sequence_file = _original_sequence_path(ziphmm_dir)
        return os.path.isfile(original_sequence_file)

    def is_cached(self, state_count):
        """
        Return True if a zipped sequence exists in the cache for the specified
        number of states; otherwise, return False.

        state_count -- The number of states in the hidden Markov model.
        """
        path = self.__resolve_zipped_sequence_path(state_count)
        return os.path.isfile(path)

    def load(self, state_count):
        """
        Load the zipped sequence for the specified number of states.  If the
        file does not exist in the cache, create it.
        """
        assert state_count > 0

        if not self.is_cached(state_count):
            return self.create_cache(state_count)

        #
        # TODO Pass copy of the substitution table.
        #
        alphabet_size = self.__zipped_alphabet_sizes[state_count]
        substitutions = self.__substitutions
        symbols_path = self.__resolve_zipped_sequence_path(state_count)
        symbols = read_symbol_array(symbols_path)
        return ZipSequence(lambda: (
            symbols,
            alphabet_size,
            state_count,
            substitutions,
            self.__unzipped_alphabet_size))

    def __resolve_zipped_sequence_path(self, state_count):
        name = '{0}.seq'.format(state_count)
        return os.path.join(self.__state_count_to_sequence_path, name)

    def __save_data_structure(self):
        lines = []

        lines.extend(['orig_alphabet_size', self.__unzipped_alphabet_size])
        lines.extend(['orig_seq_length', self.__unzipped_sequence_length])

        lines.extend(['nStates2alphabet_size'])
        for state_count in sorted(self.__zipped_alphabet_sizes):
            alphabet_size = self.__zipped_alphabet_sizes[state_count]
            lines.append('{0} {1}'.format(state_count, alphabet_size))

        lines.extend(['symbol2pair'])
        for new_symbol in sorted(self.__substitutions):
            a, b = self.__substitutions[new_symbol]
            lines.append('{0} {1} {2}'.format(new_symbol, a, b))

        with open(self.__data_structure_path, 'w') as f:
            f.write('\n'.join(map(str, lines)))
