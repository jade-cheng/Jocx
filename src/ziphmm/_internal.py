import array


def read_symbol_array(path):
    with open(path, 'r') as f:
        return array.array('H', map(int, f.read().split()))


def format_sequence(seq):
    def wrap(s):
        return '{' + s + '}'
    a, b = (seq, '') if len(seq) <= 10 else (seq[0:10], ' ...')
    return wrap(' '.join(map(str, a)) + b)


def write_symbol_array(path, symbols):
    with open(path, "w") as f:
        f.write(' '.join(map(str, symbols)))
