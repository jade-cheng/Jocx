import array

from sequence import Sequence
from _internal import format_sequence
from _internal import write_symbol_array


def note(s):
    import timeit
    print '{0}: {1}'.format(timeit.default_timer(), s)


def _find_max_adjacency(sequence, alphabet_size):
    assert alphabet_size > 0
    assert all(map(lambda x: 0 <= x < alphabet_size, sequence))

    #
    # Recreate the table; in Python, this is much faster than clearing the
    # values in the arrays.
    #
    table = [
        array.array('L', [0] * alphabet_size)
        for _ in xrange(alphabet_size)
    ]

    #
    # Count all consecutive pairs, excluding the first pair which may not be
    # compressed.
    #
    n = len(sequence)
    if n > 1:
        a = sequence[1]
        i = 2
        while i < n:
            b = sequence[i]
            table[a][b] += 1
            if a == b:
                j = i + 1
                if j < n:
                    c = sequence[j]
                    if a == c:
                        i += 1
            a = b
            i += 1

    #
    # Find the maximum value in the table.
    #
    max_count, max_pair = table[0][0], (0, 0)
    for a in xrange(alphabet_size):
        table_a = table[a]
        for b in xrange(alphabet_size):
            count = table_a[b]
            if count > max_count:
                max_count, max_pair = count, (a, b)

    return max_pair


def _pow(n, e):
    r = 1
    while e > 0:
        e -= 1
        r *= n
    return r


def _estimate_effort(count, size, zip_size, zip_length):
    #
    # NOTE This was determined by examining the number of operations required
    # for the ZipHMM Forward algorithm, expressing it as a function, and
    # simplifying that function using Mathematica.
    #
    # count      -- The number of states in the HMM.
    # size       -- The size of the unzipped alphabet.
    # zip_size   -- The size of the zipped alphabet.
    # zip_length -- The length of the zipped sequence.
    #
    t1 = count * (2 * zip_length)
    t2 = 2 * _pow(count, 3) * (zip_size - size)
    t3 = 2 * (zip_size + zip_length - (size + 1))
    t4 = _pow(count, 2) * (zip_size + (2 * (zip_length + size - 1)))
    return t1 + t2 + t3 + t4


class ZipSequence:
    """
    Implementation of a zipped sequence.  In the interest of speed, this class
    assumes ownership of the unzipped sequences provided to it.
    """

    def __init__(self, factory):
        """
        Initialize a new instance of the ZipSequence class based on the values
        provided by the specified factory. Do not expect this method to be
        executed from consumers, who should instead use the from_sequence
        method and the ZipDirectory.load method.

        factory -- The factory used to generate the class data.
        """

        x = factory()
        self.__symbols = x[0]
        self.__alphabet_size = x[1]
        self.__state_count = x[2]
        self.__substitutions = x[3]
        self.__original_alphabet_size = x[4]

    @classmethod
    def from_sequence(cls, sequence, state_count):
        """
        Initialize a new instance of the class from the specified uncompressed
        sequence and an optimal state count.

        sequence --    The sequence to compress.
        state_count -- The optimal number of states in the hidden Markov model.
        """

        assert isinstance(sequence, Sequence)
        assert len(sequence) > 0
        assert state_count > 0

        #
        # Loop until it is more effort to decompress fewer symbols of greater
        # variety.
        #
        substitutions = dict()
        symbols = array.array('H', sequence)
        alphabet_size = sequence.alphabet_size
        effort = _estimate_effort(
            state_count,
            alphabet_size,
            alphabet_size,
            len(symbols))

        while True:
            new_alphabet_size = alphabet_size + 1
            new_symbol = alphabet_size
            new_symbols = array.array('H')

            #
            # Find the pair of symbols that occurs most frequently in the
            # sequence and replace it with the new symbol; note the first
            # symbol of the sequence cannot be compressed.
            #
            (a, b) = _find_max_adjacency(symbols, alphabet_size)
            new_symbols.append(symbols[0])
            index = 1
            while index + 1 < len(symbols):
                if symbols[index] == a and symbols[index + 1] == b:
                    new_symbols.append(new_symbol)
                    index += 1
                else:
                    new_symbols.append(symbols[index])
                index += 1
            if index < len(symbols):
                new_symbols.append(symbols[index])

            #
            # Test the exit condition before adding the symbol to the
            # substitution table and swapping in the new buffer and
            # alphabet.
            #
            new_effort = _estimate_effort(
                state_count,
                sequence.alphabet_size,
                new_alphabet_size,
                len(new_symbols))

            if new_effort >= effort:
                break

            substitutions[new_symbol] = (a, b)
            alphabet_size = new_alphabet_size
            effort = new_effort
            symbols = new_symbols

        return cls(lambda: (
            symbols,
            alphabet_size,
            state_count,
            substitutions,
            sequence.alphabet_size))

    def __getitem__(self, index):
        return self.__symbols[index]

    def __len__(self):
        return len(self.__symbols)

    def __repr__(self):
        return format_sequence(self.__symbols)

    @property
    def alphabet_size(self):
        """ The number of symbols in the compressed alphabet. """
        return self.__alphabet_size

    @property
    def extended_alphabet_size(self):
        """ The number of symbols excluding the original alphabet. """
        return self.alphabet_size - self.original_alphabet_size

    @property
    def original_alphabet_size(self):
        """ The number of symbols in the uncompressed alphabet. """
        return self.__original_alphabet_size

    @property
    def state_count(self):
        """ The optimal number of states for likelihood calculations. """
        return self.__state_count

    def get_substitute(self, symbol):
        """
        Return a pair of symbols (left, right) that can be used in place of the
        specified symbol, or None if the symbol should not be replaced.

        symbol -- The symbol to substitute.
        """
        return None if symbol not in self.__substitutions \
            else self.__substitutions[symbol]

    def iterate_alphabet(self):
        """ Iterate through the symbols of the alphabet. """
        return xrange(self.alphabet_size)

    def iterate_extended_alphabet(self):
        """ Iterate through the extended symbols of the alphabet. """
        return xrange(self.original_alphabet_size, self.alphabet_size)

    def iterate_original_alphabet(self):
        """ Iterate through the original symbols of the alphabet. """
        return xrange(self.original_alphabet_size)

    def iterate_original_sequence(self):
        """ Iterate through the symbols of the original sequence. """
        substitutions = self.__substitutions

        def fn(s):
            if s not in substitutions:
                yield s
            else:
                (a, b) = substitutions[s]
                for lhs in fn(a):
                    yield lhs
                for rhs in fn(b):
                    yield rhs

        for symbol in self.__symbols:
            for original_symbol in fn(symbol):
                yield original_symbol

    def iterate_substitutes(self):
        """
        Iterate through the substitution table, yielding each value as the
        tuple, (extended_symbol, (left_symbol, right_symbol)).
        """
        for symbol in sorted(self.__substitutions):
            pair = self.__substitutions[symbol]
            yield symbol, pair

    def save(self, path):
        """
        Save the symbols of the sequence to the text file with the specified
        path.  Use whitespace to delimit the symbol numbers.

        path -- The path to the output file.
        """
        write_symbol_array(path, self.__symbols)
