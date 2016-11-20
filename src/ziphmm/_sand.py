#
# See: Sand et al. zipHMMlib: a highly optimised HMM library exploiting
# repetitions in the input to speed up the forward algorithm.
#

import math

from hidden_markov_model import HiddenMarkovModel
from matrix import Matrix
from zip_sequence import ZipSequence


class _ScaledMatrix:
    def __init__(self, scale, matrix):
        self.scale = scale
        self.matrix = matrix


def _normalize(matrix):
    sigma = matrix.compute_sum()
    lle = math.log(sigma)
    matrix.scale(1.0 / sigma)
    return lle


def _create_symbol_table(hmm, x_seq):
    assert isinstance(hmm, HiddenMarkovModel)
    assert isinstance(x_seq, ZipSequence)

    items = []
    for symbol in x_seq.iterate_original_alphabet():
        matrix = Matrix(hmm.state_count, hmm.state_count)
        for src_state in xrange(hmm.state_count):
            b = hmm.b[src_state, symbol]
            for dst_state in xrange(hmm.state_count):
                matrix[src_state, dst_state] = b * hmm.a[dst_state, src_state]

        scale = _normalize(matrix)
        items.append(_ScaledMatrix(scale, matrix))

    for symbol in x_seq.iterate_extended_alphabet():
        a, b = x_seq.get_substitute(symbol)
        item1 = items[a]
        item2 = items[b]
        matrix = item2.matrix * item1.matrix
        scale = item1.scale + item2.scale + _normalize(matrix)
        items.append(_ScaledMatrix(scale, matrix))

    return items


def evaluate(hmm, x_seq):
    """
    Return the log-likelihood of observing the specified compressed sequence
    assuming the given hidden Markov model.  If the likelihood cannot be
    determined, return None.

    hmm --   The hidden Markov model.
    x_seq -- The compressed sequence.
    """
    assert isinstance(hmm, HiddenMarkovModel)
    assert isinstance(x_seq, ZipSequence)

    try:
        table = _create_symbol_table(hmm, x_seq)
        column = Matrix(hmm.state_count, 1)

        for state in xrange(hmm.state_count):
            column[state, 0] = hmm.pi[state] * hmm.b[state, x_seq[0]]

        lle = _normalize(column)

        for symbol in x_seq[1:]:
            item = table[symbol]
            column = item.matrix * column
            lle += item.scale + _normalize(column)

        return lle

    except:
        return None
