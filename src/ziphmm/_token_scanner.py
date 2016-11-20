class TokenScanner:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.__tokens = f.read().split()
        self.__position = 0

    def __len__(self):
        return len(self.__tokens)

    @property
    def is_eof(self):
        return self.__position >= len(self.__tokens)

    @property
    def position(self):
        return self.__position

    def peek(self):
        return None if self.is_eof else self.__tokens[self.position]

    def read(self):
        token = self.peek()
        if token is not None:
            self.__position += 1
        return token

    def read_float(self):
        return float(self.read())

    def read_int(self):
        return int(self.read())

    def require(self, expected):
        actual = self.read()
        assert expected == actual, \
            'expected "{0}" but encountered "{1}"'.format(expected, actual)
