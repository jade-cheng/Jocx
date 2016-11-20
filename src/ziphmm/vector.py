import numpy


class Vector:
    """
    Implementation of a vector that holds values for hidden Markov models.
    """

    def __init__(self, length):
        """
        Initialize a new instance of the Vector class based on the specified
        length.

        length -- The length of the vector.
        """

        assert length >= 0
        self.__values = numpy.zeros(length)

    def __getitem__(self, index):
        return self.__values[index]

    def __len__(self):
        return len(self.__values)

    def __repr__(self):
        return repr(self.__values)

    def __str__(self):
        return str(self.__values)

    def __setitem__(self, index, value):
        self.__values[index] = value
