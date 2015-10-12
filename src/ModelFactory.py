from abc import ABCMeta, abstractmethod

import glob
import os
import random

from IsolationHMM import IsolationHMM
from IsolationInitialMigrationHMM import IsolationInitialMigrationHMM
from AdmixtureHMM import AdmixtureHMM
from Admixture23HMM import Admixture23HMM
from SingleAdmixtureHMM import SingleAdmixtureHMM
from SingleHMM import SingleHMM
from pyZipHMM import Forwarder, Matrix

COAL_MUL = 2.0
NUM_STATES = 5


def _to_pyziphmm_matrix(numpy_array):
    """
    Convert a numpy array to a ziphmm matrix
    :param numpy_array: The numpy array
    :return: The ziphmm matrix
    """
    ziphmm_matrix = Matrix(numpy_array.shape[0], numpy_array.shape[1])
    for i in xrange(numpy_array.shape[0]):
        for j in xrange(numpy_array.shape[1]):
            ziphmm_matrix[i, j] = numpy_array[i, j]
    return ziphmm_matrix


def _compute_likelihood(model, forwarders):
    """
    Return the likelihood value for a given model and sequences
    :param model: An HMM model
    :param forwarders: A list of ziphmm forwarder objects created from sequences
    :return: The likelihood value
    """

    t = _to_pyziphmm_matrix(model.transition_matrix)
    e = _to_pyziphmm_matrix(model.emission_matrix)
    pi = _to_pyziphmm_matrix(model.initial_distribution)

    return sum(forwarder.forward(pi, t, e) for forwarder in forwarders)


def _prepare_alignments(options, group_dir='ziphmm'):
    """
    Prepare alignments for the optimisation
    :param options: The model and optimiser options
    :param group_dir: The folder that contains the ziphmm sequence data
    :return: A list of ziphmm forwarder objects created with the sequence data
    """
    folders = glob.glob(os.path.join(os.path.join(options.exp_folder, group_dir), '*.ziphmm*'))
    return [random.choice(folders) for _ in xrange(len(folders))]


class Model(object):
    """
    An abstract class for demographic models
    """

    __metaclass__ = ABCMeta

    def likelihood(self, parameters):
        """
        Compute and return a valid likelihood for the given set of parameters.
        If any arithmetic errors occur, return negative infinity.
        :param parameters: The parameters to evaluate
        :return: The likelihood values evaluated with the given parameters
        """
        try:
            return self.raw_likelihood(parameters)
        except ArithmeticError:
            return float('-inf')

    @abstractmethod
    def raw_likelihood(self, parameters):
        """
        Compute and return a valid likelihood for a given set of parameters.
        If any errors occur, throw an exception.
        :param parameters: The parameters to evaluate
        :return: The likelihood values evaluated with the given parameters
        """
        pass


class _IsolationModel(Model):
    """
    This class sets up an isolation model, AB.
    """

    def __init__(self, options):
        super(_IsolationModel, self).__init__()
        alignments = _prepare_alignments(options, 'ziphmm_src1_admix')
        self.forwarders = [Forwarder.fromDirectory(arg) for arg in alignments]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 3
        model = IsolationHMM(parameters, NUM_STATES)
        return _compute_likelihood(model, self.forwarders)


class _Isolation3HMMModel(Model):
    """
    This class sets up an isolation model of 3 HMMs, AA, BB, and AB, using the
    composite likelihood method
    """

    def __init__(self, options):
        super(_Isolation3HMMModel, self).__init__()

        alignments_src1_admix = _prepare_alignments(options, 'ziphmm_src1_admix')
        alignments_src1_scr1 = _prepare_alignments(options, 'ziphmm_scr1_scr1')
        alignments_admix_admix = _prepare_alignments(options, 'ziphmm_admix_admix')
        self.forwarders_src1_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src1_admix]
        self.forwarders_src1_src1 = [Forwarder.fromDirectory(arg) for arg in alignments_src1_scr1]
        self.forwarders_admix_admix = [Forwarder.fromDirectory(arg) for arg in alignments_admix_admix]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 3

        coal_rate = parameters[1]
        recomb_rate = parameters[2]

        s_a = IsolationHMM(parameters, NUM_STATES)
        s_s = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)
        a_a = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)

        lle_s_a = _compute_likelihood(s_a, self.forwarders_src1_admix)
        lle_s_s = _compute_likelihood(s_s, self.forwarders_src1_src1)
        lle_a_a = _compute_likelihood(a_a, self.forwarders_admix_admix)
        return lle_s_a + lle_s_s + lle_a_a


class _IsolationInitialMigrationModel(Model):
    """
    This class sets up an isolation with initial migration model, AB.
    """

    def __init__(self, options):
        super(_IsolationInitialMigrationModel, self).__init__()

        alignments = _prepare_alignments(options, 'ziphmm_src1_admix')

        self.forwarders = [Forwarder.fromDirectory(arg) for arg in alignments]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 5
        model = IsolationInitialMigrationHMM(parameters, NUM_STATES, NUM_STATES)
        return _compute_likelihood(model, self.forwarders)


class _IsolationInitialMigration3HMMModel(Model):
    """
    This class sets up an isolation with initial migration model of 3 HMMs, AA,
    BB, and AB, using the composite likelihood method
    """

    def __init__(self, options):
        super(_IsolationInitialMigration3HMMModel, self).__init__()

        alignments_src1_admix = _prepare_alignments(options, 'ziphmm_src1_admix')
        alignments_src1_scr1 = _prepare_alignments(options, 'ziphmm_scr1_scr1')
        alignments_admix_admix = _prepare_alignments(options, 'ziphmm_admix_admix')
        self.forwarders_src1_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src1_admix]
        self.forwarders_src1_src1 = [Forwarder.fromDirectory(arg) for arg in alignments_src1_scr1]
        self.forwarders_admix_admix = [Forwarder.fromDirectory(arg) for arg in alignments_admix_admix]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 5

        coal_rate = parameters[2]
        recomb_rate = parameters[3]

        s_a = IsolationInitialMigrationHMM(parameters, NUM_STATES, NUM_STATES)
        s_s = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)
        a_a = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)

        lle_s_a = _compute_likelihood(s_a, self.forwarders_src1_admix)
        lle_s_s = _compute_likelihood(s_s, self.forwarders_src1_src1)
        lle_a_a = _compute_likelihood(a_a, self.forwarders_admix_admix)
        return lle_s_a + lle_s_s + lle_a_a


class _ThreePopAdmixModel(Model):
    """
    This class sets up a simple admixture model of 3 HMMs, AC, BC, and AB using
    the composite likelihood method
    """

    def __init__(self, options):
        super(_ThreePopAdmixModel, self).__init__()

        alignments_src1_admix = _prepare_alignments(options, 'ziphmm_src1_admix')
        alignments_src2_admix = _prepare_alignments(options, 'ziphmm_src2_admix')
        alignments_src1_src2 = _prepare_alignments(options, 'ziphmm_src1_src2')

        self.forwarders_src1_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src1_admix]
        self.forwarders_src2_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src2_admix]
        self.forwarders_src1_src2 = [Forwarder.fromDirectory(arg) for arg in alignments_src1_src2]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 5
        if parameters[-1] > 1:
            return float('-inf')

        src1_admix = AdmixtureHMM([parameters[0], parameters[1], parameters[2], parameters[3], 0, parameters[-1]], NUM_STATES, NUM_STATES)
        src2_admix = AdmixtureHMM([parameters[0], parameters[1], parameters[2], parameters[3], 1 - parameters[-1], 0], NUM_STATES, NUM_STATES)
        src1_src2 = IsolationHMM([parameters[0] + parameters[1], parameters[2], parameters[3]], NUM_STATES)

        lle1 = _compute_likelihood(src1_admix, self.forwarders_src1_admix)
        lle2 = _compute_likelihood(src2_admix, self.forwarders_src2_admix)
        lle3 = _compute_likelihood(src1_src2, self.forwarders_src1_src2)
        return lle1 + lle2 + lle3


class _ThreePopAdmix23Model(Model):
    """
    This class sets up a general admixture model of 3 HMMs, AC, BC, and AB using
    the composite likelihood method
    """

    def __init__(self, options):
        super(_ThreePopAdmix23Model, self).__init__()

        alignments_src1_admix = _prepare_alignments(options, 'ziphmm_src1_admix')
        alignments_src2_admix = _prepare_alignments(options, 'ziphmm_src2_admix')
        alignments_src1_src2 = _prepare_alignments(options, 'ziphmm_src1_src2')

        self.forwarders_src1_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src1_admix]
        self.forwarders_src2_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src2_admix]
        self.forwarders_src1_src2 = [Forwarder.fromDirectory(arg) for arg in alignments_src1_src2]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 7

        ts_r = parameters[3] + parameters[1] - parameters[2]
        if ts_r <= 0:
            return float('-inf')
        if parameters[6] < 0.0 or parameters[6] > 1.0:
            return float('-inf')

        s1_a = Admixture23HMM([parameters[0], parameters[1], parameters[3],
                               parameters[4], parameters[5], parameters[6]], NUM_STATES, NUM_STATES)

        s2_a = Admixture23HMM([parameters[0], parameters[2], ts_r,
                               parameters[4], parameters[5], 1 - parameters[6]], NUM_STATES, NUM_STATES)

        tau = parameters[0] + parameters[1] + parameters[3]

        s1_s2 = IsolationHMM([tau, parameters[4], parameters[5]], NUM_STATES)

        lle1 = _compute_likelihood(s1_a, self.forwarders_src1_admix)
        lle2 = _compute_likelihood(s2_a, self.forwarders_src2_admix)
        lle3 = _compute_likelihood(s1_s2, self.forwarders_src1_src2)
        return lle1 + lle2 + lle3


class _OnePopAdmix23Model(Model):
    """
    This class sets up a general admixture model with only the admixed
    population, CC.
    """

    def __init__(self, options):
        super(_OnePopAdmix23Model, self).__init__()

        alignments_admix_admix = _prepare_alignments(options, 'ziphmm_admix_admix')
        self.forwarders_admix_admix = [Forwarder.fromDirectory(arg) for arg in alignments_admix_admix]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 5

        iso_time = parameters[0]
        mig_time = parameters[1]
        coal_rate = parameters[2]
        recomb_rate = parameters[3]
        admix_prop = parameters[4]
        if admix_prop < 0.0 or admix_prop > 1.0:
            return float('-inf')

        a_a = SingleAdmixtureHMM([iso_time, mig_time, COAL_MUL * coal_rate, recomb_rate, admix_prop], NUM_STATES, NUM_STATES, NUM_STATES)

        return _compute_likelihood(a_a, self.forwarders_admix_admix)


class _TwoPopAdmix23Model(Model):
    """
    This class sets up a general admixture model with only the admixed
    population and one of the two source populations, AC, AA, and CC, using the
    composite likelihood method.
    """

    def __init__(self, options):
        super(_TwoPopAdmix23Model, self).__init__()

        alignments_src1_admix = _prepare_alignments(options, 'ziphmm_src1_admix')
        alignments_src1_scr1 = _prepare_alignments(options, 'ziphmm_scr1_scr1')
        alignments_admix_admix = _prepare_alignments(options, 'ziphmm_admix_admix')

        self.forwarders_src1_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src1_admix]
        self.forwarders_src1_src1 = [Forwarder.fromDirectory(arg) for arg in alignments_src1_scr1]
        self.forwarders_admix_admix = [Forwarder.fromDirectory(arg) for arg in alignments_admix_admix]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 6

        iso_time = parameters[0]
        mig_time = parameters[1] + parameters[2]
        coal_rate = parameters[3]
        recomb_rate = parameters[4]
        admix_prop = parameters[5]
        if admix_prop < 0.0 or admix_prop > 1.0:
            return float('-inf')

        s1_a = Admixture23HMM([iso_time, parameters[1], parameters[2], coal_rate, recomb_rate, admix_prop], NUM_STATES, NUM_STATES)

        s1_s1 = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)
        a_a = SingleAdmixtureHMM([iso_time, mig_time, COAL_MUL * coal_rate, recomb_rate, admix_prop], NUM_STATES, NUM_STATES, NUM_STATES)

        lle_s_a = _compute_likelihood(s1_a, self.forwarders_src1_admix)
        lle_s_s = _compute_likelihood(s1_s1, self.forwarders_src1_src1)
        lle_a_a = _compute_likelihood(a_a, self.forwarders_admix_admix)
        return lle_s_a + lle_s_s + lle_a_a


class _TwoPopAdmix23OneSampleModel(Model):
    """
    This class sets up a general admixture model with only the admixed
    population and one of the two source populations, AC.
    """

    def __init__(self, options):
        super(_TwoPopAdmix23OneSampleModel, self).__init__()

        alignments_src1_admix = _prepare_alignments(options, 'ziphmm_src1_admix')
        self.forwarders_src1_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src1_admix]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 6

        admix_prop = parameters[5]
        if admix_prop < 0.0 or admix_prop > 1.0:
            return float('-inf')

        s1_a = Admixture23HMM(parameters, NUM_STATES, NUM_STATES)

        return _compute_likelihood(s1_a, self.forwarders_src1_admix)


class _ThreePopAdmix23Model6HMM(Model):
    """
    This class sets up a general admixture model, AC, BC, AB, AA, BB, and CC,
    using the composite likelihood method
    """

    def __init__(self, options):
        super(_ThreePopAdmix23Model6HMM, self).__init__()

        alignments_src1_admix = _prepare_alignments(options, 'ziphmm_src1_admix')
        alignments_src2_admix = _prepare_alignments(options, 'ziphmm_src2_admix')
        alignments_src1_src2 = _prepare_alignments(options, 'ziphmm_src1_src2')
        alignments_src1_scr1 = _prepare_alignments(options, 'ziphmm_scr1_scr1')
        alignments_src2_src2 = _prepare_alignments(options, 'ziphmm_src2_src2')
        alignments_admix_admix = _prepare_alignments(options, 'ziphmm_admix_admix')

        self.forwarders_src1_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src1_admix]
        self.forwarders_src2_admix = [Forwarder.fromDirectory(arg) for arg in alignments_src2_admix]
        self.forwarders_src1_src2 = [Forwarder.fromDirectory(arg) for arg in alignments_src1_src2]
        self.forwarders_src1_src1 = [Forwarder.fromDirectory(arg) for arg in alignments_src1_scr1]
        self.forwarders_src2_src2 = [Forwarder.fromDirectory(arg) for arg in alignments_src2_src2]
        self.forwarders_admix_admix = [Forwarder.fromDirectory(arg) for arg in alignments_admix_admix]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 7

        iso_time = parameters[0]
        mig_time = parameters[3] + parameters[1]
        ts_r = mig_time - parameters[2]
        tau = iso_time + mig_time
        coal_rate = parameters[4]
        recomb_rate = parameters[5]
        admix_prop = parameters[6]
        if ts_r <= 0:
            return float('-inf')
        if admix_prop < 0.0 or admix_prop > 1.0:
            return float('-inf')

        s1_a = Admixture23HMM([iso_time, parameters[1], parameters[3], coal_rate, recomb_rate, admix_prop], NUM_STATES, NUM_STATES)
        s2_a = Admixture23HMM([iso_time, parameters[2], ts_r, coal_rate, recomb_rate, 1.0 - admix_prop], NUM_STATES, NUM_STATES)
        s1_s2 = IsolationHMM([tau, coal_rate, recomb_rate], NUM_STATES)

        s1_s1 = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)
        s2_s2 = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)
        a_a = SingleAdmixtureHMM([iso_time, mig_time, COAL_MUL * coal_rate, recomb_rate, admix_prop], NUM_STATES, NUM_STATES, NUM_STATES)

        lle1 = _compute_likelihood(s1_a, self.forwarders_src1_admix)
        lle2 = _compute_likelihood(s2_a, self.forwarders_src2_admix)
        lle3 = _compute_likelihood(s1_s2, self.forwarders_src1_src2)
        lle4 = _compute_likelihood(s1_s1, self.forwarders_src1_src1)
        lle5 = _compute_likelihood(s2_s2, self.forwarders_src2_src2)
        lle6 = _compute_likelihood(a_a, self.forwarders_admix_admix)
        return lle1 + lle2 + lle3 + lle4 + lle5 + lle6


class _ThreePopAdmix23Model15HMM(Model):
    """
    This class sets up a general admixture model, A1C1, A1C2, A2C1, A2C2, B1C1,
    B1C2, B2C1, B2C2, A1B1, A1B2, A2B1, A2B2, A1A2, B1B2, and C1C2, using the
    composite likelihood method
    """

    def __init__(self, options):
        super(_ThreePopAdmix23Model15HMM, self).__init__()

        alignments_src1_admix = [_prepare_alignments(options, 'ziphmm_src1_admix'),
                                 _prepare_alignments(options, 'ziphmm_src1_admix_2'),
                                 _prepare_alignments(options, 'ziphmm_src1_admix_3'),
                                 _prepare_alignments(options, 'ziphmm_src1_admix_4')]
        alignments_src2_admix = [_prepare_alignments(options, 'ziphmm_src2_admix'),
                                 _prepare_alignments(options, 'ziphmm_src2_admix_2'),
                                 _prepare_alignments(options, 'ziphmm_src2_admix_3'),
                                 _prepare_alignments(options, 'ziphmm_src2_admix_4')]
        alignments_src1_src2 = [_prepare_alignments(options, 'ziphmm_src1_src2'),
                                _prepare_alignments(options, 'ziphmm_src1_src2_2'),
                                _prepare_alignments(options, 'ziphmm_src1_src2_3'),
                                _prepare_alignments(options, 'ziphmm_src1_src2_4')]
        alignments_src1_scr1 = _prepare_alignments(options, 'ziphmm_scr1_scr1')
        alignments_src2_src2 = _prepare_alignments(options, 'ziphmm_src2_src2')
        alignments_admix_admix = _prepare_alignments(options, 'ziphmm_admix_admix')

        self.forwarders_src1_admix = [[Forwarder.fromDirectory(arg) for arg in algs] for algs in alignments_src1_admix]
        self.forwarders_src2_admix = [[Forwarder.fromDirectory(arg) for arg in algs] for algs in alignments_src2_admix]
        self.forwarders_src1_src2 = [[Forwarder.fromDirectory(arg) for arg in algs] for algs in alignments_src1_src2]
        self.forwarders_src1_src1 = [Forwarder.fromDirectory(arg) for arg in alignments_src1_scr1]
        self.forwarders_src2_src2 = [Forwarder.fromDirectory(arg) for arg in alignments_src2_src2]
        self.forwarders_admix_admix = [Forwarder.fromDirectory(arg) for arg in alignments_admix_admix]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 7

        iso_time = parameters[0]
        mig_time = parameters[3] + parameters[1]
        ts_r = mig_time - parameters[2]
        tau = iso_time + mig_time
        coal_rate = parameters[4]
        recomb_rate = parameters[5]
        admix_prop = parameters[6]
        if ts_r <= 0:
            return float('-inf')
        if admix_prop < 0.0 or admix_prop > 1.0:
            return float('-inf')

        s1_a = Admixture23HMM([iso_time, parameters[1], parameters[3], coal_rate, recomb_rate, admix_prop], NUM_STATES, NUM_STATES)
        s2_a = Admixture23HMM([iso_time, parameters[2], ts_r, coal_rate, recomb_rate, 1.0 - admix_prop], NUM_STATES, NUM_STATES)
        s1_s2 = IsolationHMM([tau, coal_rate, recomb_rate], NUM_STATES)

        s1_s1 = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)
        s2_s2 = SingleHMM([COAL_MUL * coal_rate, recomb_rate], NUM_STATES)
        a_a = SingleAdmixtureHMM([iso_time, mig_time, COAL_MUL * coal_rate, recomb_rate, admix_prop], NUM_STATES, NUM_STATES, NUM_STATES)

        lle_s1_a = [_compute_likelihood(s1_a, forwarders) for forwarders in self.forwarders_src1_admix]
        lle_s2_a = [_compute_likelihood(s2_a, forwarders) for forwarders in self.forwarders_src2_admix]
        lle_s1_s2 = [_compute_likelihood(s1_s2, forwarders) for forwarders in self.forwarders_src1_src2]
        lle_s1_s1 = [_compute_likelihood(s1_s1, self.forwarders_src1_src1)]
        lle_s2_s2 = [_compute_likelihood(s2_s2, self.forwarders_src2_src2)]
        lle_a_a = [_compute_likelihood(a_a, self.forwarders_admix_admix)]
        return sum(lle_s1_a) + sum(lle_s2_a) + sum(lle_s1_s2) + sum(lle_s1_s1) + sum(lle_s2_s2) + sum(lle_a_a)


def create(options):
    if options.model == 'iso':
        return _IsolationModel(options)

    if options.model == 'iim':
        return _IsolationInitialMigrationModel(options)

    if options.model == 'iso-3hmm':
        return _Isolation3HMMModel(options)

    if options.model == 'iim-3hmm':
        return _IsolationInitialMigration3HMMModel(options)

    if options.model == 'iim-3hmm':
        return _IsolationInitialMigrationModel(options)

    if options.model == 'admix':
        return _ThreePopAdmixModel(options)

    if options.model == 'admix23-1pop':
        return _OnePopAdmix23Model(options)

    if options.model == 'admix23-2pop':
        return _TwoPopAdmix23Model(options)

    if options.model == 'admix23-2pop-1sample':
        return _TwoPopAdmix23OneSampleModel(options)

    if options.model == 'admix23':
        return _ThreePopAdmix23Model(options)

    if options.model == 'admix23-6hmm':
        return _ThreePopAdmix23Model6HMM(options)

    if options.model == 'admix23-15hmm':
        return _ThreePopAdmix23Model15HMM(options)

    assert False, 'Unsupported model {0}'.format(options.model)
