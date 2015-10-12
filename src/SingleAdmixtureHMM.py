#!/usr/bin/env python

from scipy.linalg import expm

from HMM import *


class SingleAdmixtureHMM(HMM):
    """
    Implementation of the following admixture demographic model.
                 |        ancestral (second single)
                / \      ---------
               /   \
               \    \     migration (same CTMC as migration with zero migration rate)
                \   /
                 ---     ---------
                  |       single
                  |
                 * *
    """

    def __init__(self, parameters, no_single_states=10, no_mig_states=10, no_ancestral_states=10):
        """
        Initialise a new instance of the class.
"""
        assert len(parameters) == 5
        self.iso_time, self.mig_time, self.c, self.r, self.p = parameters

        self.no_single_states = no_single_states
        self.no_mig_states = no_mig_states
        self.no_ancestral_states = no_ancestral_states

        self.q_iso = iso_rate(self.r, self.c)
        self.q_mig = mig_rate(self.r, self.c, 0.0)
        self.q_sin = sin_rate(self.r, self.c)

        tau1 = self.iso_time
        tau2 = self.iso_time + self.mig_time

        self.sin_breaks = uniform_break_points(self.no_single_states, 0.0, tau1)
        self.mig_breaks = uniform_break_points(self.no_mig_states, tau1, tau2)
        self.anc_breaks = exp_break_points(self.no_ancestral_states, self.c, tau2)
        self.hmm_size = self.no_single_states + self.no_mig_states + self.no_ancestral_states

        self.ps = self.__get_ps()
        self.concatenated_ps = ConcatenatedPTable(self.ps, PROJECTIONS)
        jbs = JointProbTable(self.concatenated_ps, PROJECTIONS)

        self.joint_matrix = numpy.array(
            [[jbs[l, r] for r in xrange(self.hmm_size)] for l in xrange(self.hmm_size)])

        super(SingleAdmixtureHMM, self).__init__(
            self.joint_matrix,
            list(self.sin_breaks) + list(self.mig_breaks) + list(self.anc_breaks),
            self.c)

    def __admix_projection_mig(self):
        """
        Return the admixture projection matrix for an admixture event during a single population period.  The CTMC
        before this admixture event is the single CTMC.  The CTMC after this admixture event is the same as the
        two-population migration CTMC because after the admixture event coalescence can occur.
        :return: The admixture projection matrix
        """
        map_to = numpy.zeros(shape=(self.q_sin.shape[0], self.q_mig.shape[1]))
        for chunks, ix in SIN_STATES.iteritems():
            num_pieces = len(chunks)
            for i in xrange(2 ** num_pieces):
                choices = '{:08b}'.format(i)[::-1]
                prob = 1
                dst_state = []
                for j, chunk in enumerate(chunks):
                    choice = int(choices[j])
                    if choice == 1:
                        dst_state.append((2, chunk[1]))
                        prob *= 1.0 - self.p
                    else:
                        dst_state.append((1, chunk[1]))
                        prob *= self.p
                dst_state = frozenset(dst_state)
                map_to[ix, MIG_STATES[dst_state]] = prob
        return map_to

    def __get_ps(self):
        """
        Return the list of probabilities for each time slice
        :return: the list of probabilities for each time slice
        """
        to_return = [numpy.identity(self.q_sin.shape[0])]
        to_return.extend([expm(self.q_sin * self.sin_breaks[1]) for _ in xrange(len(self.sin_breaks))])
        mig_p = expm(self.q_mig * (self.mig_breaks[1] - self.mig_breaks[0]))
        to_return.extend([numpy.dot(self.__admix_projection_mig(), mig_p)])
        for _ in xrange(len(self.mig_breaks) - 1):
            to_return.append(mig_p)
        to_return.extend([expm(self.q_sin * (self.anc_breaks[i + 1] - self.anc_breaks[i]))
                         for i in
                         xrange(len(self.anc_breaks) - 1)])
        p_end = numpy.zeros(shape=self.q_sin.shape)
        p_end[:, END[PS.single][0]] = 1.0
        to_return.append(p_end)
        return to_return


def main():
    """
    Test main
    """
    model = SingleAdmixtureHMM([0.0001, 0.0001, 1200.0, 0.4, 0.2], 2, 2, 2)
    print model.emission_matrix
    print model.transition_matrix
    print model.initial_distribution

if __name__ == '__main__':
    main()
