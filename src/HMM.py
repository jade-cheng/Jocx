from abc import ABCMeta
from math import exp
from scipy.stats import expon, uniform
from Resources import *


def _categorize_states(states):
    """
    Return states in three groups: begin, left, and end
    :param states: The group of states to be categorized
    :return: Three groups: begin, left and end
    """
    left = {}
    end = {}
    begin = {}
    for s, i in states.iteritems():
        left_coal = False
        right_coal = False
        for p in s:
            l, r = p[1]
            if len(l) == 2:
                left_coal = True
            if len(r) == 2:
                right_coal = True
        if left_coal:
            if right_coal:
                end[i] = s
                continue
            left[i] = s
            continue
        if not right_coal:
            begin[i] = s
    return begin, left, end

# Categorize states for different models
ISO_BEGIN, _, __ = _categorize_states(ISO_STATES)
MIG_BEGIN, MIG_LEFT, MIG_END = _categorize_states(MIG_STATES)
SIN_BEGIN, SIN_LEFT, SIN_END = _categorize_states(SIN_STATES)
BUDDY23_BEGIN, ___, ____ = _categorize_states(BUDDY23_STATES)
GREEDY1_BEGIN, GREEDY1_LEFT, GREEDY1_END = _categorize_states(GREEDY1_STATES)

# Record all begin states
BEGIN = {
    4: ISO_BEGIN.keys(),
    15: SIN_BEGIN.keys(),
    94: MIG_BEGIN.keys(),
    12: BUDDY23_BEGIN.keys(),
    29: GREEDY1_BEGIN.keys()
}

# Record all left states
LEFT = {
    15: SIN_LEFT.keys(),
    94: MIG_LEFT.keys(),
    29: GREEDY1_LEFT.keys()
}

# Record all end states
END = {
    15: SIN_END.keys(),
    94: MIG_END.keys(),
    29: GREEDY1_END.keys()
}


class PS(object):
    """
    Enum for CTMCs
    """
    isolation = 4
    migration = 94
    single = 15
    buddy23 = 12
    greedy1 = 29


def get_0b(p):
    """
    Return the slice of matrix corresponding to going from the initial state
    (in isolation CTMC) to a begin state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert right in BEGIN
    if left == PS.isolation:
        return p[[[3]], BEGIN[right]]
    if left == PS.single:
        return p[[[6]], BEGIN[right]]


def get_0l(p):
    """
    Return the slice of matrix corresponding to going from the initial state
    (in isolation CTMC) to a left state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert right in LEFT
    if left == PS.isolation:
        return p[[[3]], LEFT[right]]
    if left == PS.single:
        return p[[[6]], LEFT[right]]


def get_0e(p):
    """
    Return the slice of matrix corresponding to going from the initial state
    (in isolation CTMC) to a end state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert right in END
    if left == PS.isolation:
        return p[[[3]], END[right]]
    if left == PS.single:
        return p[[[6]], END[right]]


def get_bb(p):
    """
    Return the slice of matrix going from a begin state to a begin state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert left in BEGIN
    assert right in BEGIN
    return p[[[e] for e in BEGIN[left]], BEGIN[right]]


def get_be(p):
    """
    Return the slice of matrix going from a begin state to a end state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert left in BEGIN
    assert right in END
    return p[[[e] for e in BEGIN[left]], END[right]]


def get_ll(p):
    """
    Return the slice of matrix going from a left state to a left state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert left in LEFT
    assert right in LEFT
    return p[[[e] for e in LEFT[left]], LEFT[right]]


def get_bl(p):
    """
    Return the slice of matrix going from a begin state to a left state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert left in BEGIN
    assert right in LEFT
    return p[[[e] for e in BEGIN[left]], LEFT[right]]


def get_le(p):
    """
    Return the slice of matrix going from a left state to an end state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert left in LEFT
    assert right in END
    return p[[[e] for e in LEFT[left]], END[right]]


def get_ee(p):
    """
    Return the slice of matrix going from an end state to an end state
    :param p: The matrix to slice
    :return: The matrix slice
    """
    left, right = p.shape
    assert left in END
    assert right in END
    return p[[[e] for e in END[left]], END[right]]

# Compute the projection matrix going from an isolation CTMC to a single CTMC
PROJ_ISO_SIN = numpy.zeros(shape=(iso_rate(1, 1).shape[1], sin_rate(1, 1).shape[0]))
for state, index in ISO_STATES.iteritems():
    PROJ_ISO_SIN[index, SIN_STATES[frozenset([(0, piece[1]) for piece in state])]] = 1.0

# Compute the projection matrix going from an isolation CTMC to a migration CTMC
PROJ_ISO_MIG = numpy.zeros(shape=(iso_rate(1, 1).shape[1], mig_rate(1, 1, 1).shape[0]))
for state, index in ISO_STATES.iteritems():
    PROJ_ISO_MIG[index, MIG_STATES[state]] = 1.0

# Compute the projection matrix going from an migration CTMC to a single CTMC
PROJ_MIG_SIN = numpy.zeros(shape=(mig_rate(1, 1, 1).shape[1], sin_rate(1, 1).shape[0]))
for state, index in MIG_STATES.iteritems():
    PROJ_MIG_SIN[index, SIN_STATES[frozenset([(0, piece[1]) for piece in state])]] = 1.0

# Compute the various of sliced projection matrices
PROJ_LL_MIG_SIN = get_ll(PROJ_MIG_SIN)
PROJ_BB_ISO_SIN = get_bb(PROJ_ISO_SIN)
PROJ_BB_ISO_MIG = get_bb(PROJ_ISO_MIG)
PROJ_BB_MIG_SIN = get_bb(PROJ_MIG_SIN)

# Record various of projection matrices
PROJECTIONS = {PROJ_ISO_SIN.shape: PROJ_ISO_SIN,
               PROJ_ISO_MIG.shape: PROJ_ISO_MIG,
               PROJ_MIG_SIN.shape: PROJ_MIG_SIN,
               PROJ_LL_MIG_SIN.shape: PROJ_LL_MIG_SIN,
               PROJ_BB_ISO_SIN.shape: PROJ_BB_ISO_SIN,
               PROJ_BB_ISO_MIG.shape: PROJ_BB_ISO_MIG,
               PROJ_BB_MIG_SIN.shape: PROJ_BB_MIG_SIN}


def _truncated_exp_midpoint(t1, t2, rate):
    """
    Return the mean coalescence point between t1 and t2 from a truncated
    exponential distribution.
    :param t1: Beginning of the time slice
    :param t2: Ending of the time slice
    :param rate: coalescent rate in this time slice
    :return: The exponential distribution midpoint of this time slice
    """
    delta_t = t2 - t1
    return t1 + 1.0 / rate - (delta_t * exp(-delta_t * rate)) / (1 - exp(-delta_t * rate))


def _coalescence_points(break_points, coal_rate):
    """
    Return the mean coalescence times between each time break point and after
    the last break point.
    :param break_points: Break points between the HMM states.
    :param coal_rate: The coalescent rate
    :return: the mean coalescence times between each time break point and after the last break point.
    """
    result = []
    for i in xrange(1, len(break_points)):
        t = _truncated_exp_midpoint(break_points[i - 1], break_points[i], coal_rate)
        result.append(t)
    result.append(break_points[-1] + 1.0 / coal_rate)
    return result


def _jukes_cantor(is_same, t):
    """
    Compute the Jukes-Cantor transition probability of switching in an interval
    :param is_same: A boolean indicating whether two symbols are the same
    :param t: The time slice
    :return: The probability of switching in the interval
    """
    if is_same:
        return 0.25 + 0.75 * exp(-4.0 / 3.0 * t)
    return 0.75 - 0.75 * exp(-4.0 / 3.0 * t)


def _emission_matrix(coal_points):
    """
    Return the emission matrix given the time slices and coalescent rate
    :param coal_points: A list of coalescence points to emit from
    :return: The emission matrix
    """
    emission_probabilities = numpy.zeros(shape=(len(coal_points), 3))
    for s in xrange(len(coal_points)):
        emission_probabilities[s, 0] = _jukes_cantor(True, 2 * coal_points[s])
        emission_probabilities[s, 1] = _jukes_cantor(False, 2 * coal_points[s])
        emission_probabilities[s, 2] = 1.0
    return emission_probabilities


def exp_break_points(no_intervals, coal_rate, offset=0.0):
    """
    Return break points for equal probably intervals over exponential
    distribution given the coalescent rate
    :param no_intervals: The number of intervals to make
    :param coal_rate: The coalescent rate
    :param offset: An offset added to all break points
    :return: A list of break points
    """
    points = expon.ppf([float(i) / no_intervals for i in xrange(no_intervals)])
    return points / coal_rate + offset


def uniform_break_points(no_intervals, start, end):
    """
    Return uniformly distributed break points
    :param start: The start of the interval
    :param end: The end of the interval
    :return: A list of break points
    """
    points = uniform.ppf([float(i) / no_intervals for i in xrange(no_intervals)])
    return points * (end - start) + start


class ConcatenatedPTable(object):
    """
    Encapsulate concatenated probability table
    """

    def __init__(self, ps, projections):
        """
        Initialise a new instance of the class.
        :param ps: The list of probabilities for each time period
        :param projections: The projections that glue together probabilities
                            from different time periods
        """
        self.__ps = ps
        self.__sequence = {}
        self.__projections = projections

    @property
    def size(self):
        """
        Return the size of the collection
        :return: the size of the collection
        """
        return len(self.__sequence)

    def __getitem__(self, ix):
        """
        Retrieve a probability or compute it if it's not already computed
        :param ix: The row and column indices
        :return: The concatenated probability at the row and column indices
        """
        row, col = ix
        assert col >= row

        def multiply_ps(p1, p2):
            lhs = p1.shape[1]
            rhs = p2.shape[0]
            if lhs == rhs:
                return numpy.dot(p1, p2)
            assert (lhs, rhs) in self.__projections
            return numpy.dot(numpy.dot(p1, self.__projections[(lhs, rhs)]), p2)

        if ix not in self.__sequence:
            self.__sequence[ix] = self.__ps[row] if row == col else \
                multiply_ps(self[row, col - 1], self.__ps[col])

        # print ix, self.__sequence[ix]
        return self.__sequence[ix]


class JointProbTable(object):
    """
    Encapsulate the table of joint probabilities
    """

    def __init__(self, concatenated_ps, projections):
        """
        Initialise a new instance of the class.
        :param concatenated_ps: The table of concatenated probabilities
        :param projections: The projections that glue together probabilities
                            from different time slices
        """
        self.__concatenated_ps = concatenated_ps
        self.__joint_probs = {}
        self.__projections = projections

    def __getitem__(self, ix):
        """
        Retrieve a probability or compute it if it's not already computed
        :param ix: The row and column indices
        :return: The joint probability at the row and column indices
        """
        if ix in self.__joint_probs:
            return self.__joint_probs[ix]

        l, r = ix
        if l > r:
            return self[r, l]

        def multiply_ps(p1, p2):
            if p1 is None:
                return p2
            if p2 is None:
                return p1
            left = p1.shape[1]
            right = p2.shape[0]
            if left == right:
                return numpy.dot(p1, p2)
            assert (left, right) in self.__projections
            return numpy.dot(numpy.dot(p1, self.__projections[(left, right)]), p2)

        to_sum = get_0b(self.__concatenated_ps[(0, l)])
        if l == r:
            self.__joint_probs[ix] = multiply_ps(to_sum, get_be(self.__concatenated_ps[l + 1, l + 1])).sum()
        else:
            to_sum = multiply_ps(to_sum, get_bl(self.__concatenated_ps[l + 1, l + 1]))
            if r >= l + 2:
                to_sum = multiply_ps(to_sum, get_ll(self.__concatenated_ps[(l + 2, r)]))
            to_sum = multiply_ps(to_sum, get_le(self.__concatenated_ps[r + 1, r + 1]))
            self.__joint_probs[ix] = to_sum.sum()

        return self.__joint_probs[ix]


class HMM(object):
    """
    An abstract class of HHMs for different demographic models
    """

    __metaclass__ = ABCMeta

    def __init__(self, joint_matrix, breakpoints, coal_rate):
        """
        Initialize a HMM given a joint probability matrix, a list of
        breakpoints, and a coalescence rate.
        :param joint_matrix: The joint probability matrix
        :param breakpoints: The list of breakpoints
        :param coal_rate: The coalescence rate
        :return: A new instance
        """
        hmm_size = len(breakpoints)

        self.__emission_matrix = numpy.asmatrix(_emission_matrix(_coalescence_points(breakpoints, coal_rate)))
        self.__initial_distribution = numpy.asmatrix(joint_matrix.sum(axis=1)).T
        row_sums = joint_matrix.sum(axis=1)
        j_rows = []
        for l in xrange(hmm_size):
            factor = 0 if row_sums[l] <= 0.0 else 1.0 / row_sums[l]
            j_rows.append([factor * joint_matrix[l, r] for r in xrange(hmm_size)])
        self.__transition_matrix = numpy.asmatrix(j_rows)

    @property
    def emission_matrix(self):
        """
        Return the HMM emission probability matrix
        :return: The HMM emission probability matrix
        """
        return self.__emission_matrix

    @property
    def initial_distribution(self):
        """
        Return the HMM initial distribution
        :return: The HMM initial distribution
        """
        return self.__initial_distribution

    @property
    def transition_matrix(self):
        """
        Return the HMM transition probability matrix
        :return: The HMM transition probability matrix
        """
        return self.__transition_matrix
