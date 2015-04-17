from abc import ABCMeta, abstractmethod

import glob
import os
import random

from IsolationHMM import IsolationHMM
from IsolationInitialMigrationHMM import IsolationInitialMigrationHMM
from AdmixtureHMM import AdmixtureHMM
from Admixture23HMM import Admixture23HMM
from pyZipHMM import Forwarder, Matrix


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
    This class sets up an isolation model
    """

    def __init__(self, options):
        super(_IsolationModel, self).__init__()
        alignments = _prepare_alignments(options)
        self.forwarders = [Forwarder.fromDirectory(arg) for arg in alignments]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 3
        model = IsolationHMM(parameters)
        return _compute_likelihood(model, self.forwarders)


class _IsolationInitialMigrationModel(Model):
    """
    This class sets up an isolation with initial migration model
    """

    def __init__(self, options):
        super(_IsolationInitialMigrationModel, self).__init__()

        alignments = _prepare_alignments(options)

        self.forwarders = [Forwarder.fromDirectory(arg) for arg in alignments]

    def raw_likelihood(self, parameters):
        assert len(parameters) == 5
        model = IsolationInitialMigrationHMM(parameters)
        return _compute_likelihood(model, self.forwarders)


class _ThreePopAdmixModel(Model):
    """
    This class sets up a simple admixture model using the composite likelihood method
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

        src1_admix = AdmixtureHMM([parameters[0], parameters[1], parameters[2], parameters[3], 0, parameters[-1]])
        src2_admix = AdmixtureHMM([parameters[0], parameters[1], parameters[2], parameters[3], 1 - parameters[-1], 0])
        src1_src2 = IsolationHMM([parameters[0] + parameters[1], parameters[2], parameters[3]])

        lle1 = _compute_likelihood(src1_admix, self.forwarders_src1_admix)
        lle2 = _compute_likelihood(src2_admix, self.forwarders_src2_admix)
        lle3 = _compute_likelihood(src1_src2, self.forwarders_src1_src2)
        return lle1 + lle2 + lle3


class _ThreePopAdmix23Model(Model):
    """
    This class sets up a admixture model with two different speciation times
    using the composite likelihood method
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

        s1_a = Admixture23HMM([parameters[0], parameters[1], parameters[3],
                               parameters[4], parameters[5], parameters[6]])

        s2_a = Admixture23HMM([parameters[0], parameters[2], ts_r,
                              parameters[4], parameters[5], 1 - parameters[6]])

        tau = parameters[0] + parameters[1] + parameters[3]

        s1_s2 = IsolationHMM([tau, parameters[4], parameters[5]])

        lle1 = _compute_likelihood(s1_a, self.forwarders_src1_admix)
        lle2 = _compute_likelihood(s2_a, self.forwarders_src2_admix)
        lle3 = _compute_likelihood(s1_s2, self.forwarders_src1_src2)
        return lle1 + lle2 + lle3


def create(options):
    if options.model == 'iso':
        return _IsolationModel(options)

    if options.model == 'iim':
        return _IsolationInitialMigrationModel(options)

    if options.model == 'admix':
        return _ThreePopAdmixModel(options)

    if options.model == 'admix23':
        return _ThreePopAdmix23Model(options)

    assert False, 'Unsupported model {0}'.format(options.model)
