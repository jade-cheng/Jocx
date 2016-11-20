#
# See: Rabiner L. R. 'A Tutorial on Hidden Markov Models and Selected
# Applications in Speech Recognition'. Proc. IEEE, Vol. 77, No. 2, 1989.
#

from hidden_markov_model import HiddenMarkovModel
from matrix import Matrix
from sequence import Sequence


def _compute_alpha(hmm, seq):
    assert isinstance(hmm, HiddenMarkovModel)
    assert isinstance(seq, Sequence)

    n = hmm.state_count
    t_max = len(seq)
    alpha = Matrix(n, t_max)

    for i in xrange(n):
        alpha[i, 0] = hmm.pi[i] * hmm.b[i, seq[0]]

    for t in xrange(t_max - 1):
        for j in xrange(n):
            sigma = 0.0
            for i in xrange(n):
                sigma += alpha[i, t] * hmm.a[i, j]
            alpha[j, t + 1] = sigma * hmm.b[j, seq[t + 1]]

    return alpha


def _evaluate_alpha(alpha):
    assert isinstance(alpha, Matrix)
    sigma = 0.0
    t_max = alpha.width - 1
    for row in xrange(alpha.height):
        sigma += alpha[row, t_max]
    return sigma


def evaluate(hmm, seq):
    """
    Return the likelihood of observing the specified sequence assuming the
    given hidden Markov model.

    hmm --  The hidden Markov model.
    seq -- The sequence.
    """
    assert isinstance(hmm, HiddenMarkovModel)
    assert isinstance(seq, Sequence)
    return _evaluate_alpha(_compute_alpha(hmm, seq))
