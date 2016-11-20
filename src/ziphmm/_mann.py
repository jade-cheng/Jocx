#
# See: Mann T. P. 'Numerically stable hidden markov model implementation'.
#

import math
import numpy

from hidden_markov_model import HiddenMarkovModel
from matrix import Matrix
from sequence import Sequence


def _isnan(x):
    return numpy.isnan(x)


def _eexp(x):
    if _isnan(x):
        return 0.0
    return math.exp(x)


def _eln(x):
    if x == 0.0:
        return numpy.nan
    assert x > 0.0, x
    return math.log(x)


def _elnproduct(elnx, elny):
    if _isnan(elnx):
        return numpy.nan
    if _isnan(elny):
        return numpy.nan
    return elnx + elny


def _elnsum(elnx, elny):
    if _isnan(elnx):
        return elny
    if _isnan(elny):
        return elnx
    if elnx > elny:
        return elnx + _eln(1.0 + math.exp(elny - elnx))
    else:
        return elny + _eln(1.0 + math.exp(elnx - elny))


def _compute_alpha(hmm, seq):
    assert isinstance(hmm, HiddenMarkovModel)
    assert isinstance(seq, Sequence)

    n = hmm.state_count
    t_max = len(seq)
    alpha = Matrix(n, t_max)

    for i in xrange(n):
        alpha[i, 0] = _elnproduct(
            _eln(hmm.pi[i]),
            _eln(hmm.b[i, seq[0]]))

    for t in xrange(t_max - 1):
        for j in xrange(n):
            sigma = numpy.nan

            for i in xrange(n):
                sigma = _elnsum(
                    sigma,
                    _elnproduct(
                        alpha[i, t],
                        _eln(hmm.a[i, j])))

            alpha[j, t + 1] = _elnproduct(
                sigma,
                _eln(hmm.b[j, seq[t + 1]]))

    return alpha


def _evaluate_alpha(alpha):
    assert isinstance(alpha, Matrix)
    sigma = numpy.nan
    t_max = alpha.width - 1
    for row in xrange(alpha.height):
        sigma = _elnsum(sigma, alpha[row, t_max])
    return None if _isnan(sigma) else sigma


def evaluate(hmm, seq):
    """
    Return the log-likelihood of observing the specified sequence assuming the
    given hidden Markov model.  If the likelihood cannot be determined, return
    None.

    hmm -- The hidden Markov model.
    seq -- The sequence.
    """
    assert isinstance(hmm, HiddenMarkovModel)
    assert isinstance(seq, Sequence)
    return _evaluate_alpha(_compute_alpha(hmm, seq))
