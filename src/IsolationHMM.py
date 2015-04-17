#!/usr/bin/env python

from scipy.linalg import expm

from HMM import *


class IsolationHMM(HMM):
    """
    Implementation of the isolation demographic model.
    """

    def __init__(self, parameters, no_ancestral_states=10):
        """
        Initialise a new instance of the class.
        :param parameters: The split time, coalescent rate, and recombination rate
        :param no_ancestral_states: The number of time slices in the isolation period
        """
        assert len(parameters) == 3
        self.tau, self.coal_rate, self.recomb_rate = parameters
        self.no_ancestral_states = no_ancestral_states
        self.q_iso = iso_rate(self.recomb_rate, self.coal_rate)
        self.q_sin = sin_rate(self.recomb_rate, self.coal_rate)
        self.sin_breaks = exp_break_points(self.no_ancestral_states, self.coal_rate, self.tau)
        self.ps = self.__get_ps()
        self.concatenated_ps = ConcatenatedPTable(self.ps, PROJECTIONS)

        jbs = JointProbTable(self.concatenated_ps, PROJECTIONS)
        self.joint_matrix = numpy.array(
            [[jbs[l, r] for r in xrange(self.no_ancestral_states)] for l in xrange(self.no_ancestral_states)])

        super(IsolationHMM, self).__init__(
            self.joint_matrix,
            list(self.sin_breaks),
            self.coal_rate)

    def __get_ps(self):
        """
        Return the list of probabilities for each time slice
        :return: the list of probabilities for each time slice
        """
        to_return = [expm(self.q_iso * self.sin_breaks[0])]
        to_return.extend([expm(self.q_sin * (self.sin_breaks[i + 1] - self.sin_breaks[i]))
                         for i in
                         xrange(len(self.sin_breaks) - 1)])
        p_end = numpy.zeros(shape=self.q_sin.shape)
        p_end[:, END[PS.single][0]] = 1.0
        to_return.append(p_end)
        return to_return


def main():
    """
    Test main that constructs and prints an isolation model
    """
    model = IsolationHMM([0.002, 2000.0, 0.8], 3)
    print model.emission_matrix
    print model.transition_matrix
    print model.initial_distribution


if __name__ == '__main__':
    main()
