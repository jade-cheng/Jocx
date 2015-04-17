#!/usr/bin/env python

from scipy.linalg import expm

from HMM import *


class IsolationInitialMigrationHMM(HMM):
    """
    Implementation of the isolation with initial migration demographic model.
    """

    def __init__(self, parameters, no_mig_states=10, no_ancestral_states=10):
        """
        Initialise a new instance of the class.
        :param parameters: The isolation time, migration, coalescent rate, recombination rate, and migration rate
        :param no_mig_states: The number of time slices in the migration period
        :param no_ancestral_states: The number of time slices in the ancestral (single population) period
        """
        assert len(parameters) == 5
        self.iso_time, self.mig_time, self.coal_rate, self.recomb_rate, self.mig_rate = parameters
        self.no_mig_states = no_mig_states
        self.no_ancestral_states = no_ancestral_states
        self.q_iso = iso_rate(self.recomb_rate, self.coal_rate)
        self.q_mig = mig_rate(self.recomb_rate, self.coal_rate, self.mig_rate)
        self.q_sin = sin_rate(self.recomb_rate, self.coal_rate)
        tau1 = self.iso_time
        tau2 = self.iso_time + self.mig_time
        self.mig_breaks = uniform_break_points(self.no_mig_states, tau1, tau2)
        self.sin_breaks = exp_break_points(self.no_ancestral_states, self.coal_rate, tau2)
        self.hmm_size = self.no_mig_states + self.no_ancestral_states
        self.ps = self.__get_ps()
        self.concatenated_ps = ConcatenatedPTable(self.ps, PROJECTIONS)

        jbs = JointProbTable(self.concatenated_ps, PROJECTIONS)
        self.joint_matrix = numpy.array(
            [[jbs[l, r] for r in xrange(self.hmm_size)] for l in xrange(self.hmm_size)])

        super(IsolationInitialMigrationHMM, self).__init__(
            self.joint_matrix,
            list(self.mig_breaks) + list(self.sin_breaks),
            self.coal_rate)

    def __get_ps(self):
        """
        Return the list of probabilities for each time slice
        :return: the list of probabilities for each time slice
        """
        to_return = [expm(self.q_iso * self.mig_breaks[0])]
        mig_p = expm(self.q_mig * (self.mig_breaks[1] - self.mig_breaks[0]))
        to_return.extend([mig_p for _ in xrange(len(self.mig_breaks))])
        to_return.extend([expm(self.q_sin * (self.sin_breaks[i + 1] - self.sin_breaks[i]))
                         for i in
                         xrange(len(self.sin_breaks) - 1)])
        p_end = numpy.zeros(shape=self.q_sin.shape)
        p_end[:, END[PS.single][0]] = 1.0
        to_return.append(p_end)
        return to_return


def main():
    """
    Test main that constructs and prints an isolation with initial migration model
    """
    model = IsolationInitialMigrationHMM([0.5, 1.0, 1, 0.4, 0.001], 3, 3)
    print model.emission_matrix
    print model.transition_matrix
    print model.initial_distribution


if __name__ == '__main__':
    main()
