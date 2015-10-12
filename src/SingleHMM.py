#!/usr/bin/env python

from scipy.linalg import expm

from HMM import *


class SingleHMM(HMM):
    """
    Implementation of the following demographic model.
          |
          |     single
         * *
    """

    def __init__(self, parameters, no_single_states=10):
        """
        Initialise a new instance of the class.
        """
        assert len(parameters) == 2
        self.c, self.r = parameters

        self.no_single_states = no_single_states

        self.q_sin = sin_rate(self.r, self.c)

        self.sin_breaks = exp_break_points(self.no_single_states, self.c, 0.0)
        self.hmm_size = self.no_single_states

        self.ps = self.__get_ps()
        self.concatenated_ps = ConcatenatedPTable(self.ps, PROJECTIONS)
        jbs = JointProbTable(self.concatenated_ps, PROJECTIONS)

        self.joint_matrix = numpy.array(
            [[jbs[l, r] for r in xrange(self.hmm_size)] for l in xrange(self.hmm_size)])

        super(SingleHMM, self).__init__(self.joint_matrix, list(self.sin_breaks), self.c)

    def __get_ps(self):
        """
        Return the list of probabilities for each time slice
        :return: the list of probabilities for each time slice
        """
        to_return = [numpy.identity(self.q_sin.shape[0])]
        to_return.extend([expm(self.q_sin * (self.sin_breaks[i + 1] - self.sin_breaks[i]))
                         for i in
                         xrange(len(self.sin_breaks) - 1)])
        p_end = numpy.zeros(shape=self.q_sin.shape)
        p_end[:, END[PS.single][0]] = 1.0
        to_return.append(p_end)
        return to_return


def main():
    """
    Test main
    """
    model = SingleHMM([0.4, 0.2], 10)
    print model.emission_matrix
    print model.transition_matrix
    print model.initial_distribution

if __name__ == '__main__':
    main()
