class Vector2d:
    def __init__(self, value =[]):
        self._value = list(value)

    def shape(self):
        return len(self._value),

    def same_shape(self, other):
        if self.shape() == other.shape():
            return
        else:
            raise ShapeError

    def __add__(self, other):
        self.same_shape(other)
        return Vector2d([x + y for x, y in zip(self._value, other._value)])

    def __eq__(self, other):
        self.same_shape(other)
        return self._value == other._value

    def __sub__(self, other):
        self.same_shape(other)
        return Vector2d([x - y for x, y in zip(self._value, other._value)])

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            raise TypeError

    def dot(self, other):
        self.same_shape(other)
        return sum([x * y for x, y in zip(self._value, other._value) ])

    def __mul__(self, other):
        return Vector2d([x * other for x in self._value])


    def mean(*args):
        shapes = [x.shape() for x in args]
        if len(set(shapes)) == 1:
            arg_vals = [x._value for x in args]
            return Vector2d([sum(x) / len(x) for x in zip(*arg_vals)])
        else:
            raise ShapeError

    def magnitude(self):
        return sum([x ** 2 for x in self._value]) ** 0.5

class Matrix:
    def __init__(self, value = [[]]):
        self._value = value

    def shape(self):
        columns = [len(x) for x in self._value]
        if len(set(columns)) != 1:
            raise ShapeError
        else:
            return (len(self._value), list(set(columns))[0])

    def row(self, index):
        return self._value[index][:]

    def col(self, index):
        return self.get_transpose().row(index)

    def get_transpose(self):
        return Matrix([list(x) for x in zip(*self._value)])

# def matrix_add(matrix1, matrix2):
#     if shape(matrix1) == shape(matrix2):
#         return [vector_add(x, y) for x, y in zip(matrix1, matrix2)]
#     else:
#         raise ShapeError
#
# def matrix_sub(matrix1, matrix2):
#     if shape(matrix1) == shape(matrix2):
#         return [vector_sub(x, y) for x, y in zip(matrix1, matrix2)]
#     else:
#         raise ShapeError
#
# def matrix_scalar_multiply(matrix, scalar):
#     return [vector_multiply(x,scalar) for x in matrix]
#
# def matrix_vector_multiply(matrix, vector):
#     if shape(matrix)[1] == shape(vector)[0]:
#         return [dot(x, vector) for x in matrix]
#     else:
#         raise ShapeError
#

# def matrix_matrix_multiply(matrix1, matrix2):
#     if shape(matrix1)[1] == shape(matrix2) [0]:
#         transposed_product = [[dot(m1, m2) for m1 in matrix1] for m2 in zip(*matrix2)]
#         return transpose_matrix(transposed_product)
#     else:
#         raise ShapeError
#         return


class ShapeError(Exception):
    pass

def shape(matrix):
    try:
        columns = [len(x) for x in matrix]
    except TypeError:
        return (len(matrix),)
    if len(set(columns)) != 1:
        raise ShapeError
    else:
        return (len(matrix), list(set(columns))[0])

def vector_add(vector1, vector2):
    if shape(vector1) == shape(vector2):
        return [x + y for x, y in zip(vector1, vector2)]
    else:
        raise ShapeError

def vector_sub(vector1, vector2):
    if shape(vector1) == shape(vector2):
        return[x - y for x, y in zip(vector1, vector2)]
    else:
        raise ShapeError

def vector_sum(*args):
    shapes = [shape(x) for x in args]
    if len(set(shapes)) == 1:
        return [ sum(x) for x in zip(*args)]
    else:
        raise ShapeError

def dot(vector1, vector2):
    if shape(vector1) == shape(vector2):
        return(sum([x * y for x, y in zip(vector1, vector2) ]))
    else:
        raise ShapeError

def vector_multiply(vector, scalar):
    return [x * scalar for x in vector]

def vector_mean(*args):
    shapes = [shape(x) for x in args]
    if len(set(shapes)) == 1:
        return [sum(x) / len(x) for x in zip(*args)]
    else:
        raise ShapeError

def magnitude(vector):
    return sum([x ** 2 for x in vector]) ** 0.5

def matrix_row(matrix, index):
    return matrix[index]

def matrix_col(matrix, index):
    return matrix_row(transpose_matrix(matrix),index)

def matrix_add(matrix1, matrix2):
    if shape(matrix1) == shape(matrix2):
        return [vector_add(x, y) for x, y in zip(matrix1, matrix2)]
    else:
        raise ShapeError

def matrix_sub(matrix1, matrix2):
    if shape(matrix1) == shape(matrix2):
        return [vector_sub(x, y) for x, y in zip(matrix1, matrix2)]
    else:
        raise ShapeError

def matrix_scalar_multiply(matrix, scalar):
    return [vector_multiply(x,scalar) for x in matrix]

def matrix_vector_multiply(matrix, vector):
    if shape(matrix)[1] == shape(vector)[0]:
        return [dot(x, vector) for x in matrix]
    else:
        raise ShapeError

def transpose_matrix(matrix):
    return [list(x) for x in zip(*matrix)]

def matrix_matrix_multiply(matrix1, matrix2):
    if shape(matrix1)[1] == shape(matrix2) [0]:
        transposed_product = [[dot(m1, m2) for m1 in matrix1] for m2 in zip(*matrix2)]
        return transpose_matrix(transposed_product)
    else:
        raise ShapeError
