#!/usr/bin/env python

import os
import sys

import ModelFactory
import OptimiserFactory


USAGE = """USAGE:
  python {name} <ziphmm-dir> <model> <optimiser>
  python {name} --help

ARGUMENTS
  ziphmm-dir   The path to the ziphmm data directory
  model        The model of choice: iso, iim, admix, or admix23
  optimiser    The optimiser of choice: nm, ga, pso, or dea
  --help       Prints this help message

DESCRIPTION
  This program executes CoalHMM with a specified model and optimiser using a
  given sequence of data in the format of ziphmm directories.  The program
  prints to standard output the progression of the estimated parameters and the
  corresponding log likelihood.

EXAMPLES
  $ python {name} ./path/to/ziphmm iso ga
  # algorithm            = _GAOptimiser
  # timeout              = None
  # elite_count          = 1
  :
  # gen idv            fitness          param0          param1          param2
      1   1    -38786.09710120      0.00083995    164.91715870      0.01344298
      1   2    -59912.38796110      0.00098375     31.17524014      0.36805062
  :

BUGS
  Report all bugs to Jade Cheng <ycheng@cs.au.dk>

Copyright (c) 2015 Jade Cheng
"""


class Options(object):
    def __init__(self, args):
        # Path to the directory that contains the ziphmm sub-directories
        self.exp_folder = args[1]

        # Model to execute, such as 'iso' and 'iim'
        self.model = args[2]

        # Optimizer to use, such as 'pso' and 'ga'
        self.opt_choice = args[3]

        # This information determines the bounds for the optimisers. Each
        # value in this table has two bits of data:
        #   bounds_mul   the range with respect to the center points
        #   targets      the center points for the legal parameter ranges
        target_options = {
            'iso':                  (10, [float(e) for e in args[4:][0:3]]),
            'iim':                  (10, [float(e) for e in args[4:][0:5]]),

            'iso-3hmm':             (10, [float(e) for e in args[4:][0:3]]),
            'iim-3hmm':             (10, [float(e) for e in args[4:][0:5]]),

            'admix':                (10, [float(e) for e in args[4:][0:5]]),
            'admix23-1pop':         (10, [float(e) for e in args[4:][0:5]]),

            'admix23-2pop':         (10, [float(e) for e in args[4:][0:6]]),
            'admix23-2pop-1sample': (10, [float(e) for e in args[4:][0:6]]),

            'admix23':              (10, [float(e) for e in args[4:][0:7]]),
            'admix23-6hmm':         (10, [float(e) for e in args[4:][0:7]]),
            'admix23-15hmm':        (10, [float(e) for e in args[4:][0:7]])

        }

        assert self.model in target_options, 'unsupported model type {0}'.format(self.model)
        self.bounds_mul, self.targets = target_options[self.model]


def main():

    if '--help' in sys.argv:
        print USAGE.format(name=os.path.basename(sys.argv[0]))
        exit()

    options = Options(sys.argv)

    model = ModelFactory.create(options)

    optimiser = OptimiserFactory.create(options)

    optimiser.maximise(model.likelihood)

if __name__ == '__main__':
    main()
