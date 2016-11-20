from _internal import format_sequence
from _internal import read_symbol_array
from _internal import write_symbol_array


class Sequence:
    """
    Implementation of a sequence of observations.  Each observation is
    represented as an integer value from zero to one less than the alphabet
    size of the sequence.
    """

    def __init__(self, symbols, alphabet_size=None):
        """
        Initialize a new instance of the Sequence class based on the specified
        symbols and, optionally, alphabet size.  Assume ownership of the list
        of symbols.  If the alphabet size is not provided, determine the
        alphabet size as one more than the largest value in the list of
        symbols.  Otherwise, require all symbols to range between zero and one
        less than the alphabet size, inclusive.  In addition, require at least
        one symbol to exist in the list.

        symbols --       The symbols of the sequence.
        alphabet_size -- (Optional) The size of the alphabet.
        """

        assert len(symbols) > 0
        assert all(map(lambda x: x >= 0, symbols))

        self.__symbols = symbols

        if alphabet_size is None:
            self.__alphabet_size = max(symbols) + 1
        else:
            self.__alphabet_size = alphabet_size
            assert alphabet_size >= max(symbols) + 1

    def __getitem__(self, index):
        return self.__symbols[index]

    def __len__(self):
        return len(self.__symbols)

    def __repr__(self):
        return format_sequence(self.__symbols)

    @property
    def alphabet_size(self):
        """ The size of the alphabet. """
        return self.__alphabet_size

    def iterate_alphabet(self):
        """ Iterate through the symbols of the alphabet. """
        return xrange(self.__alphabet_size)

    def save(self, path):
        """
        Save the symbols of the sequence to the text file with the specified
        path.  Use whitespace to delimit the symbol numbers.

        path -- The path to the output file.
        """
        write_symbol_array(path, self.__symbols)

    @classmethod
    def from_file(cls, path, alphabet_size=None):
        """
        Initialize a new instance based on the contents of the specified text
        file and, optionally, alphabet size.  If the alphabet size is not
        provided, the alphabet size is determined as one more than the largest
        value in the list of symbols.  Otherwise, all symbols must range
        between zero and one less than the alphabet size, inclusive.  At least
        one symbol must be present in the file.

        path --          The path to the file to read.
        alphabet_size -- (Optional) The size of the alphabet.
        """
        symbols = read_symbol_array(path)
        return cls(symbols, alphabet_size)
