import os
import shutil
import tempfile
import uuid
import ziphmm


def create_hmm(resource_name):
    return ziphmm.HiddenMarkovModel.from_file(locate(resource_name))


def create_seq(resource_name):
    return ziphmm.Sequence.from_file(locate(resource_name))


def create_ziphmm_directory(parent_directory):
    ziphmm_directory = os.path.join(parent_directory, str(uuid.uuid4()))
    os.mkdir(ziphmm_directory)

    xseq_directory = os.path.join(ziphmm_directory, 'nStates2seq')
    os.mkdir(xseq_directory)

    shutil.copyfile(
        locate('inputs/ziphmm/data_structure'),
        os.path.join(ziphmm_directory, 'data_structure'))

    shutil.copyfile(
        locate('inputs/ziphmm/original_sequence'),
        os.path.join(ziphmm_directory, 'original_sequence'))

    shutil.copyfile(
        locate('inputs/ziphmm/nStates2seq/2.seq'),
        os.path.join(xseq_directory, '2.seq'))

    return ziphmm_directory


def format_matrix(m):
    row = []
    for y in xrange(m.height):
        col = []
        for x in xrange(m.width):
            col.append(str(m[y, x]))
        row.append('{' + ','.join(col) + '}')
    return '{' + ','.join(row) + '}'


def format_vector(v):
    return '{' + ','.join(map(str, v)) + '}'


def locate(resource_name):
    return os.path.join(os.path.dirname(
        os.path.realpath(__file__)), resource_name)


def seq_to_string(seq):
    return ' '.join(map(str, seq))


class TempDirectory:
    def __init__(self):
        self.__path = tempfile.mkdtemp()

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        shutil.rmtree(self.__path, True)
        return False

    @property
    def path(self):
        return self.__path
