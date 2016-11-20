#!/usr/bin/env python

import os
import sys

import ModelFactory
import OptimiserFactory


USAGE = """USAGE:
  python {name} init [options] <exp-folder> <model> <fasta1> <fasta2> ...
  python {name} run <exp-folder> <model> <optimizer> <range1> <range2> ...

COMMANDS
  init            Initializes alignments and zipped sequences needed to run an
                  experiment. This command must be executed once before using
                  the 'run' command.
  run             Runs an experiment. Before running an experiment, the
                  appropriate directories must be initialized using the 'init'
                  command.

OPTIONS
  --chunk-size    Indicates the next argument is the maximum size of an
                  individual alignment produced by the ZipHMM library; units
                  are in symbols. If unspecified, the chunk size is set to the
                  size of the entire sequence.
  --help          Prints this help message and exits

ARGUMENTS
  exp-folder      The path to the experiments folder
  fasta1          The path to the first FASTA-formatted file
  fasta2          The path to the second FASTA-formatted file
  model           The model to use for the experiment; e.g. 'iso' or 'iim'
  optimizer       The optimizer to use for the experiment; e.g. 'nm' or 'pso'
  range1          The range for the first parameter (see README for details)
  range2          The range for the second parameter

DESCRIPTION
  This program executes CoalHMM with a specified model and optimizer using a
  given sequence of data in the format of ZipHMM directories.  The program
  prints to standard output the progression of the estimated parameters and
  the corresponding log likelihood.

  ISOLATION MODEL (iso)
      *
     / \\ tau
    A   B
    3 params -> tau, coal_rate, recomb_rate
    2 seqs   -> A, B
    1 group  -> AB

  ISOLATION INITIAL MIGRATION MODEL (iim)
        *
       / \\
      /<->\\    mig_time
     /     \\   iso_time
    A       B
    5 params -> iso_time, mig_time, coal_rate, recomb_rate, mig_rate
    2 seqs   -> A, B
    1 group  -> AB

  ISOLATION 3 HMM MODEL (iso-3hmm)
      *
     / \\ tau
    A1  B1
    A2  B2
    3 params -> tau, coal_rate, recomb_rate
    4 seqs   -> A1, A2, B1, B2
    3 groups -> A1B1, A1A2, B1B2

  ISOLATION INITIAL MIGRATION 3 HMM MODEL (iim-3hmm)
        *
       / \\
      /<->\\   mig_time
     /     \\  iso_time
    A1      B1
    A2      B2
    5 params -> iso_time, mig_time, coal_rate, recomb_rate, mig_rate
    4 seqs   -> A1, A2, B1, B2
    3 groups -> A1B1, A1A2, B1B2

  THREE POP ADMIX MODEL (admix)
                   *
                  / \\
                 /   \\    mig_time
                /_____\\
    admix_prop / <-|   \\  iso_time
              A    C    B
    5 params -> iso_time, mig_time, coal_rate, recomb_rate, admix_prop
    3 seqs   -> A, B, C
    3 groups -> AC, BC, AB

  THREE POP ADMIX 2 3 MODEL (admix23)
                      *
                     / \\     greedy1_time_1a
    buddy23_time_1a /\\  \\
                   /  \\_/\\   buddy23_time_2a
       admix_prop /  <-|  \\  iso_time
                 A    C    B
    7 params -> iso_time, buddy23_time_1a, buddy23_time_2a, greedy1_time_1a,
                coal_rate, recomb_rate, admix_prop
    3 seqs   -> A, B, C
    3 groups -> AC, BC, AB

  ONE POP ADMIX 2 3 MODEL (admix23-1pop)
                      *
                     / \\     greedy1_time_1a
    buddy23_time_1a /\\  \\
                   /  \\_/\\   buddy23_time_2a
       admix_prop /  <-|  \\  iso_time
                      C1
                      C2
    5 params -> iso_time, mig_time, coal_rate, recomb_rate, admix_prop
    2 seqs   -> C1, C2
    1 groups -> C1C2

  TWO POP ADMIX 2 3 MODEL (admix23-2pop)
                      *
                     / \\     greedy1_time_1a
    buddy23_time_1a /\\  \\
                   /  \\_/\\   buddy23_time_2a
       admix_prop /  <-|  \\  iso_time
                 A1   C1
                 A2   C2
    6 params -> iso_time, buddy23_time, greedy1_time, coal_rate, recomb_rate,
                admix_prop
    4 seqs   -> A1, A2, C1, C2
    3 groups -> A1C1, A1A2, C1C2

  TWO POP ADMIX 2 3 ONE SAMPLE MODEL (admix23-2pop-1sample)
                      *
                     / \\     greedy1_time_1a
    buddy23_time_1a /\\  \\
                   /  \\_/\\   buddy23_time_2a
       admix_prop /  <-|  \\  iso_time
                 A    C
    6 params -> iso_time, buddy23_time, greedy1_time, coal_rate, recomb_rate,
                admix_prop
    2 seqs   -> A, C
    1 group  -> AC

  THREE POP ADMIX 2 3 MODEL 6 HMM (admix23-6hmm)
                      *
                     / \\     greedy1_time_1a
    buddy23_time_1a /\\  \\
                   /  \\_/\\   buddy23_time_2a
       admix_prop /  <-|  \\  iso_time
                 A1   C1   B1
                 A2   C2   B2
    7 params -> iso_time, buddy23_time_1a, buddy23_time_2a, greedy1_time_1a,
                coal_rate, recomb_rate, admix_prop
    6 seqs   -> A1, A2, B1, B2, C1, C2
    6 groups -> A1C1, B1C1, A1B1, A1A2, B1B2, C1C2

  THREE POP ADMIX 2 3 MODEL 15 HMM (admix23-15hmm)
                      *
                     / \\     greedy1_time_1a
    buddy23_time_1a /\\  \\
                   /  \\_/\\   buddy23_time_2a
       admix_prop /  <-|  \\  iso_time
                 A1   C1   B1
                 A2   C2   B2
    7 params -> iso_time, buddy23_time_1a, buddy23_time_2a, greedy1_time_1a,
                coal_rate, recomb_rate, admix_prop
    6 seqs   -> A1, A2, B1, B2, C1, C2
    15 groups-> A1C1, A1C2, A2C1, A2C2, B1C1
                B1C2, B2C1, B2C2, A1B1, A1B2
                A2B1, A2B2, A1A2, B1B2, C1C2

EXAMPLES
  $ python {name} init .../ziphmm iso .../seq1.fa .../seq2.fa
  $ python {name} run  .../ziphmm iso ga
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

Copyright (c) 2015-2016 Jade Cheng
"""


class RunOptions:
    def __init__(self, args):
        assert len(args) >= 3, 'invalid syntax; try --help'

        self.exp_folder = args.pop(0)
        self.model = args.pop(0)
        self.opt_choice = args.pop(0)
        self.bounds_mul = 2.0
        self.targets = map(float, args)

        model = ModelFactory.create(self)
        assert len(args) == model.param_count, \
            'invalid syntax; try --help'


class InitOptions:
    def __init__(self, args):
        assert len(args) >= 4, 'invalid syntax; try --help'

        self.chunk_size = None
        while '--chunk-size' in args:
            index = args.index('--chunk_size')
            assert len(args) > index, 'invalid syntax; try --help'
            args.pop(index)
            self.chunk_size = int(args.pop(index))
            assert self.chunk_size > 0, 'invalid chunk size'

        self.exp_folder = args.pop(0)
        self.model = args.pop(0)
        self.fasta = args

        model = ModelFactory.create(self)
        assert len(self.fasta) == model.fasta_count, \
            'invalid syntax; try --help'


def _init(args):
    options = InitOptions(args)
    model = ModelFactory.create(options)
    model.init_alignments()


def _run(args):
    options = RunOptions(args)
    model = ModelFactory.create(options)
    model.load_alignments()
    optimiser = OptimiserFactory.create(options)
    optimiser.maximise(model.likelihood)


def main():

    if '--help' in sys.argv:
        print USAGE.format(name=os.path.basename(sys.argv[0]))
        exit()

    assert len(sys.argv) >= 2, 'invalid syntax; try --help'
    command = sys.argv[1]

    if command == 'init':
        _init(sys.argv[2:])
        return

    if command == 'run':
        _run(sys.argv[2:])
        return

    assert False, 'invalid command; try --help'

if __name__ == '__main__':
    main()
