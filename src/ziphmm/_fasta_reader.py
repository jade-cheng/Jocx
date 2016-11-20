_DEFAULT_ALPHABET = 'acgturykmswbdhvnx-'


class _FastaEntry:
    def __init__(self, position):
        self.position = position
        self.length = 0

    def __str__(self):
        return '({0},{1})'.format(self.position, self.length)


class FastaReader:
    """
    Implementation of a reader of FASTA files.
    """
    def __init__(self, path):
        """
        Initialize a new instance of the FastaReader class by reading the file
        with the specified path. Raise an IOError if the file does not parse
        successfully.

        path -- The path to the FASTA file.
        """

        self.__current = None
        self.__handle = open(path, 'r')
        self.__entries = dict()
        self.__path = path
        self.__parse()
        self.__names = tuple(sorted(self.__entries.keys()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__handle.close()
        return False

    def __str__(self):
        return str(self.__names)

    @property
    def current_sequence(self):
        """ The name of the current sequence being read, or None. """
        return self.__current

    @property
    def names(self):
        """ The tuple of names for the sequences of the FASTA file. """
        return self.__names

    @property
    def path(self):
        """ The path to the FASTA file. """
        return self.__path

    def get_length(self, name):
        """
        Return the number of symbols in the specified sequence. Raise a
        LookupError if the sequence is not found.

        name -- The name of the sequence.
        """
        return self.__find(name).length

    def read_symbol(self):
        """
        Read and return the next symbol from the current sequence. If no more
        symbols are available, return None.
        """
        while True:
            ch = self.__handle.read(1)
            if not ch:
                return None
            if ch == '>':
                self.__handle.seek(0, 2)
                return None
            if not ch.isspace():
                return ch

    def seek(self, name):
        """
        Seek to the start of the specified sequence. Raise a LookupError if
        the sequence does not exist.

        name -- The name of the sequence.
        """
        self.__seek(self.__find(name))
        self.__current = name

    def validate(self, name, alphabet=None):
        """
        Validate the specified sequence contains only symbols in the specified
        alphabet. If unspecified, use a default alphabet. If any symbol is
        invalid, raise a RuntimeError. If the sequence is not found, raise a
        LookupError.

        name --     The name of the sequence to validate.
        alphabet -- The alphabet, or None to use a default alphabet.
        """

        entry = self.__find(name)
        self.__validate(entry, alphabet)

    def validate_all(self, alphabet=None):
        """
        Validate all sequences contain only symbols in the specified alphabet.
        If unspecified, use a default alphabet. If any symbol is invalid,
        raise a RuntimeError.

        alphabet -- The alphabet, or None to use a default alphabet.
        """
        for name in self.__names:
            self.validate(name, alphabet)

    def __find(self, name):
        if name not in self.__entries:
            raise LookupError((
                'The sequence name "{0}" does not exist in the FASTA file ' +
                '"{1}"').format(name, self.__path))
        return self.__entries[name]

    def __parse(self):
        entry = None

        while True:
            ch = self.__handle.read(1)
            if not ch:
                break

            if ch.isspace():
                continue

            if ch == '>':
                name = self.__handle.readline().strip()
                if name in self.__entries:
                    raise IOError((
                        'Duplicate sequence name "{0}" in FASTA file "' +
                        '{0}"').format(name, self.__path))

                entry = _FastaEntry(self.__handle.tell())
                self.__entries[name] = entry
                continue

            if entry is None:
                raise IOError((
                    'Encountered symbol "{0}" before ">" in FASTA file ' +
                    '"{1}"').format(ch, self.__path))

            entry.length += 1

        self.__handle.seek(0, 2)

    def __seek(self, entry):
        self.__handle.seek(entry.position)

    def __validate(self, entry, alphabet):
        if alphabet is None:
            alphabet = _DEFAULT_ALPHABET

        flags = [False for _ in xrange(255)]
        for ch in alphabet:
            flags[ord(ch.lower())] = True
            flags[ord(ch.upper())] = True

        self.__seek(entry)

        while True:
            ch = self.read_symbol()
            if not ch:
                break
            if not flags[ord(ch)]:
                raise RuntimeError((
                    'Invalid symbol "{0}" found at offset {1} of FASTA ' +
                    'file "{2}" with alphabet "{3}"').format(
                    ch,
                    self.__handle.tell() - 1,
                    self.__path,
                    alphabet))
