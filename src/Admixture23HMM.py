#!/usr/bin/env python

from scipy.linalg import expm

from HMM import *


class Admixture23HMM(HMM):
    """
    Implementation of an admixture demographic model. The composite likelihood method is used with two admixture model
    to form the following demographic
                 |        sin
     ----       / \
               /   \      greedy1
     ----     /\    \
             /  \   /\    buddy23
     ----   /    ---  \
           /      |    \  2-pop iso
     ---- /       |     \
    """

    def __init__(self, parameters, no_mid_states=10, no_ancestral_states=10):
        """
        Initialise a new instance of the class.
        :param parameters: The isolation time, the time after the admixture event and before the first speciation
                            event, the time between two speciation events, coalescent rate, recombination rate, and
                            the admixture proportion from the admixed population
        :param no_mid_states: The number of time slices between the two speciation times
        :param no_ancestral_states: The number of time slices in the ancestral (single population) period
        """
        self.iso_time, self.buddy23_time, self.greedy1_time, self.coal_rate, self.r, self.p = parameters
        self.no_mid_states = no_mid_states
        self.no_ancestral_states = no_ancestral_states

        self.q_iso = iso_rate(self.r, self.coal_rate)
        self.q_buddy23 = buddy23_rate(0, 0, self.coal_rate, self.coal_rate, self.coal_rate, self.r)
        self.q_greedy1 = greedy1_rate(0, self.coal_rate, self.coal_rate, self.r)
        self.q_sin = sin_rate(self.r, self.coal_rate)

        self.ta = self.iso_time
        self.tm = self.ta + self.buddy23_time
        self.ts = self.tm + self.greedy1_time
        self.greedy1_breaks = uniform_break_points(self.no_mid_states, self.tm, self.ts)
        self.sin_breaks = exp_break_points(self.no_ancestral_states, self.coal_rate, self.ts)
        self.hmm_size = self.no_ancestral_states + self.no_mid_states

        self.iso_to_buddy23 = self.__admix_projection()
        self.buddy23_to_greedy1 = self.__buddy23_to_greedy1()
        self.greedy1_to_sin = self.__greedy1_to_sin()
        self.buddy23_to_greedy1_bb = get_bb(self.buddy23_to_greedy1)
        self.greedy1_to_sin_bb = get_bb(self.greedy1_to_sin)
        self.greedy1_to_sin_ll = get_ll(self.greedy1_to_sin)
        self.projections = {
            self.buddy23_to_greedy1.shape: self.buddy23_to_greedy1,
            self.greedy1_to_sin.shape: self.greedy1_to_sin,
            self.buddy23_to_greedy1_bb.shape: self.buddy23_to_greedy1_bb,
            self.greedy1_to_sin_bb.shape: self.greedy1_to_sin_bb,
            self.greedy1_to_sin_ll.shape: self.greedy1_to_sin_ll
        }

        self.ps = self.__get_ps()

        self.concatenated_ps = ConcatenatedPTable(self.ps, self.projections)
        jbs = JointProbTable(self.concatenated_ps, self.projections)

        self.joint_matrix = numpy.array(
            [[jbs[l, r] for r in xrange(self.hmm_size)] for l in xrange(self.hmm_size)])

        super(Admixture23HMM, self).__init__(
            self.joint_matrix,
            list(self.greedy1_breaks) + list(self.sin_breaks),
            self.coal_rate)

    def __admix_projection(self):
        """
        Return the admixture projection matrix for an admixture event during a two-population isolation period.  In
        this admixture event, the admixed population may send lineages into a third population. The CTMC before this
        admixture event is the two-population isolation CTMC.  The CTMC after this admixture event is the buddy23 CTMC.
        :return: The admixture projection matrix
        """
        map_to = numpy.zeros(shape=(self.q_iso.shape[1], self.q_buddy23.shape[0]))
        for chunks, i in ISO_STATES.iteritems():
            pop1_chunks = []
            pop2_chunks = []
            for chunk in chunks:
                if chunk[0] == 2:
                    pop2_chunks.append(chunk)
                else:
                    pop1_chunks.append(chunk)
            for j in xrange(2 ** len(pop2_chunks)):
                prob = 1
                dst_state = list(pop1_chunks)
                choices = '{:08b}'.format(j)[::-1]
                for k, pop2_chunk in enumerate(pop2_chunks):
                    choice = int(choices[k])
                    if choice == 1:
                        dst_state.append(pop2_chunk)
                        prob *= self.p
                    else:
                        dst_state.append((3, pop2_chunk[1]))
                        prob *= 1 - self.p
                map_to[i, BUDDY23_STATES[frozenset(dst_state)]] = prob
        return map_to

    def __buddy23_to_greedy1(self):
        """
        Return the projection matrix for an population structure change from a two-population isolation period to a
        single ancestral population.  The initial two-population isolation may have some lineages residing in a third
        population.  We call this scenario buddy23.  The CTMC before this speciation event is the buddy23 CTMC.  The
        CTMC after this speciation event is the single CTMC.
        :return: The projection matrix
        """
        map_to = numpy.zeros(shape=(self.q_buddy23.shape[1], self.q_greedy1.shape[0]))
        for chunks, i in BUDDY23_STATES.iteritems():
            dst_state = []
            for chunk in chunks:
                if chunk[0] == 1:
                    dst_state.append(chunk)
                elif chunk[0] == 2:
                    dst_state.append((1, chunk[1]))
                else:
                    dst_state.append((2, chunk[1]))
            map_to[i, GREEDY1_STATES[frozenset(dst_state)]] = 1
        return map_to

    def __greedy1_to_sin(self):
        map_to = numpy.zeros(shape=(self.q_greedy1.shape[1], self.q_sin.shape[0]))
        for chunks, i in GREEDY1_STATES.iteritems():
            dst_state = [(0, chunk[1]) for chunk in chunks]
            map_to[i, SIN_STATES[frozenset(dst_state)]] = 1
        return map_to

    def __get_ps(self):
        """
        Return the list of probabilities for each time slice
        :return: the list of probabilities for each time slice
        """
        p_iso = expm(self.q_iso * self.iso_time)
        p_buddy23 = expm(self.q_buddy23 * self.buddy23_time)
        to_return = [numpy.dot(numpy.dot(p_iso, self.iso_to_buddy23), p_buddy23)]
        p_greedy1 = expm(self.q_greedy1 * (self.greedy1_breaks[1] - self.greedy1_breaks[0]))
        to_return.extend([p_greedy1 for _ in xrange(len(self.greedy1_breaks))])
        for i in xrange(len(self.sin_breaks) - 1):
            p_sin = expm(self.q_sin * (self.sin_breaks[i + 1] - self.sin_breaks[i]))
            to_return.append(p_sin)
        p_end = numpy.zeros(shape=self.q_sin.shape)
        p_end[:, END[PS.single][0]] = 1.0
        to_return.append(p_end)
        return to_return


def main():
    """
    Test main that constructs and prints an Admixture23 model
    """
    model = Admixture23HMM([0.0001, 0.0001, 0.0001, 1000, 0.4, 0.7], 2, 2)
    print model.initial_distribution
    print model.transition_matrix
    print model.emission_matrix

    # write_model(Admix23([0.001, 0.001, 0.009, 1200, 0.4, 0.9]), 'src1_admix')
    # write_model(Admix23([0.001, 0.002, 0.008, 1200, 0.4, 0.1]), 'src2_admix')

if __name__ == '__main__':
    main()
