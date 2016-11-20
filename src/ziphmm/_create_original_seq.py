import os
import sys

from _fasta_reader import FastaReader


def _get_shared_seq_names(fasta1, fasta2, reader1, reader2):
    assert isinstance(reader1, FastaReader)
    assert isinstance(reader2, FastaReader)

    names = set()
    skipped = set()

    def add_name(ra, rb, fb):
        for n in ra.names:
            if n in skipped:
                continue
            if n not in rb.names:
                sys.stderr.write((
                    '# WARNING: skipping sequence "{0}" because it does ' +
                    'not exist in "{1}"\n').format(n, fb))
            elif ra.get_length(n) != rb.get_length(n):
                skipped.add(n)
                sys.stderr.write((
                    '# WARNING: skipping sequence "{0}" because it has ' +
                    'inconsistent lengths {1} != {2}\n').format(
                        n,
                        ra.get_length(n),
                        rb.get_length(n)))
            else:
                names.add(n)

    add_name(reader1, reader2, fasta2)
    add_name(reader2, reader1, fasta1)
    return sorted(names)


def create_original_sequence(ziphmm_dir, fasta1, fasta2, chunk_size=None, logger=None):
    """
    Create the original_sequence file of a ZipHMM directory based on the data
    from the two specified FASTA files. Store the sequence into the specified
    output directory. If a sequence from the FASTA files is larger than the
    specified chunk size, then break the sequence into multiple directories,
    each containing up to the chunk size number of symbols. If a sequence
    exists in only one of the two FASTA files or if it has an inconsistent
    length across the two files, skip it (and write a warning to stderr). If
    any other error occurs, raise an IOError or RuntimeError.

    ziphmm_dir -- The output directory.
    fasta1 --     The path to the first FASTA file.
    fasta2 --     The path to the second FASTA file.
    chunk_size -- The chunk size, which must be greater than zero or None.
    logger --     The output logger, or None to run silently.
    """

    assert chunk_size is None or chunk_size > 0

    def write(s):
        if logger is not None:
            logger.write(s)
            logger.write('\n')

    write('# creating uncompressed sequence file')
    write('# using output directory "{0}"'.format(ziphmm_dir))

    if chunk_size is not None:
        write('# using chunk size {0}'.format(chunk_size))

    write('# parsing "{0}"'.format(fasta1))
    with FastaReader(fasta1) as reader1:

        write('# parsing "{0}"'.format(fasta2))
        with FastaReader(fasta2) as reader2:
            for name in _get_shared_seq_names(fasta1, fasta2, reader1, reader2):
                seq_length = reader1.get_length(name)

                if chunk_size is None:
                    num_chunks = 1
                else:
                    num_chunks = int(seq_length / chunk_size)
                    if 0 != seq_length % chunk_size:
                        num_chunks += 1

                write('# comparing sequence "{0}"'.format(name))
                write('# sequence length: {0}'.format(seq_length))
                if chunk_size is not None:
                    write('# number of chunks to create: {0}'.format(num_chunks))
                sym_count = seq_length if chunk_size is None else chunk_size

                reader1.seek(name)
                reader2.seek(name)

                for j in xrange(num_chunks):
                    ziphmm_subdir = os.path.join(
                        ziphmm_dir, '{0}.ziphmm'.format(name))
                    if chunk_size is not None:
                        ziphmm_subdir += str(j)

                    seq_path = os.path.join(ziphmm_subdir, 'original_sequence')
                    if os.path.isfile(seq_path):
                        write('# file already exist: {0}'.format(seq_path))
                        continue

                    write('# creating "{0}"'.format(ziphmm_subdir))
                    os.mkdir(ziphmm_subdir)

                    with open(seq_path, 'w') as out:
                        for k in xrange(sym_count):
                            ch1 = reader1.read_symbol()
                            ch2 = reader2.read_symbol()
                            if not ch1:
                                if not ch2:
                                    break
                                raise IOError((
                                    'unexpected end of stream in file ' +
                                    '"{0}"').format(fasta1))
                            if not ch2:
                                raise IOError((
                                    'unexpected end of stream in file ' +
                                    '"{0}"').format(fasta2))

                            if ch1 == '-' and ch2 == '-':
                                out.write('2 ')
                            elif ch1 != ch2:
                                out.write('1 ')
                            else:
                                out.write('0 ')
