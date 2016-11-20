import numpy


class Matrix:
    """
    Implementation of a matrix that holds values for hidden Markov models.
    """

    def __init__(self, height, width):
        """
        Initialize a new instance of the Matrix class based on the specified
        width and height.  Allow one dimension to equal to zero if any only if
        the other dimension is also equal to zero.

        height -- The number of rows in the matrix.
        width --  The number of columns in the matrix.
        """

        assert (height == width == 0) or (height > 0 and width > 0)

        self.__height = height
        self.__width = width
        self.__values = numpy.zeros((height, width))

    def __getitem__(self, (row, column)):
        return self.__values[row, column]

    def __setitem__(self, (row, column), value):
        self.__values[row, column] = value

    def __mul__(self, other):
        out = Matrix(0, 0)
        out.__values = numpy.dot(self.__values, other.__values)
        out.__height, out.__width = out.__values.shape
        return out

    def __repr__(self):
        return repr(self.__values)

    def __str__(self):
        return str(self.__values)

    @property
    def height(self):
        """ The number of rows in the matrix. """
        return self.__height

    @property
    def width(self):
        """ The number of columns in the matrix. """
        return self.__width

    def compute_sum(self):
        """ Compute the sum of all values in the matrix. """
        return self.__values.sum()

    def scale(self, value):
        """
        Scale all values in the matrix by the specified value.

        value -- The scaling factor.
        """
        self.__values *= value
