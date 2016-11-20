from vector import Vector
from matrix import Matrix

from _token_scanner import TokenScanner


class HiddenMarkovModel:
    """
    Implementation of a hidden Markov model, a statistical Markov model in
    which the system being modeled is assumed to be a Markov process with
    unobserved (hidden) states.
    """

    def __init__(self, state_count, alphabet_size):
        """
        Initialize a new instance of the HiddenMarkovModel class based on the
        specified number of states and alphabet size.

        state_count --   The number of states; this must be a positive value.
        alphabet_size -- The alphabet size; this must be a positive value.
        """

        assert state_count > 0
        assert alphabet_size > 0

        self.__state_count = state_count
        self.__alphabet_size = alphabet_size

        self.__pi = Vector(state_count)
        self.__a = Matrix(state_count, state_count)
        self.__b = Matrix(state_count, alphabet_size)

    @property
    def a(self):
        """
        The state transition probability distribution.  Rows indicate a source
        state, and columns indicate a destination state.  For example,
        a[1,2] indicates the probability of leaving state 1 to enter state 2.
        """
        return self.__a

    @property
    def alphabet_size(self):
        """ The alphabet size. """
        return self.__alphabet_size

    @property
    def b(self):
        """
        The observation symbol probability distribution.  Rows indicate a
        state, and columns indicate a symbol.  For example, b[1,2] indicates
        the probability of being in state 1 and observing symbol 2.
        """
        return self.__b

    @property
    def pi(self):
        """
        The initial state distribution.  This is a column vector where rows
        indicate a state.  For example, pi[3] indicates the initial probability
        that the model is in state 3.
        """
        return self.__pi

    @property
    def state_count(self):
        """ The number of states. """
        return self.__state_count

    def save(self, path):
        """
        Save this instance to the specified text file.

        path -- The path to the file to save.
        """

        with open(path, "w") as f:
            lines = []

            def append_vector(title, vector):
                lines.append(title)
                for n in vector:
                    lines.append(n)

            def append_matrix(title, matrix):
                lines.append(title)
                for y in xrange(matrix.height):
                    lines.append(' '.join(map(str, [
                        matrix[y, x] for x in xrange(matrix.width)])))

            lines.extend(['no_states', self.__state_count])
            lines.extend(['alphabet_size', self.__alphabet_size])
            append_vector('pi', self.__pi)
            append_matrix('A', self.__a)
            append_matrix('B', self.__b)
            f.write('\n'.join(map(str, lines)))

    @classmethod
    def from_file(cls, path):
        """
        Initialize a new instance of the HiddenMarkovModel class based on the
        contents of the specified text file.

        path -- The path to the file to load.
        """

        scanner = TokenScanner(path)

        scanner.require('no_states')
        state_count = scanner.read_int()

        scanner.require('alphabet_size')
        alphabet_size = scanner.read_int()

        hmm = cls(state_count, alphabet_size)

        scanner.require('pi')
        for j in xrange(state_count):
            hmm.pi[j] = scanner.read_float()

        scanner.require('A')
        for i in xrange(state_count):
            for j in xrange(state_count):
                hmm.a[i, j] = scanner.read_float()

        scanner.require('B')
        for i in xrange(state_count):
            for j in xrange(alphabet_size):
                hmm.b[i, j] = scanner.read_float()

        assert scanner.is_eof
        return hmm
