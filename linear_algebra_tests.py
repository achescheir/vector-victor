# # Code from here was provided from:
# # https://github.com/tiyd-python-2016-02/assignments/blob/master/week2/linear_algebra_test.py
import math
from linear_algebra import *
from nose.tools import raises


def are_equal(x, y, tolerance=0.001):
    """Helper function to compare floats, which are often not quite equal."""
    return abs(x - y) <= tolerance


m = [3, 4]
n = [5, 0]

v = [1, 3, 0]
w = [0, 2, 4]
u = [1, 1, 1]
y = [10, 20, 30]
z = [0, 0, 0]


def test_shape_vectors():
    """shape takes a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    assert shape([1]) == (1,)
    assert shape(m) == (2,)
    assert shape(v) == (3,)


def test_vector_add():
    """
    [a b]  + [c d]  = [a+c b+d]
    Matrix + Matrix = Matrix
    """
    assert vector_add(v, w) == [1, 5, 4]
    assert vector_add(u, y) == [11, 21, 31]
    assert vector_add(u, z) == u


def test_vector_add_is_commutative():
    assert vector_add(w, y) == vector_add(y, w)


@raises(ShapeError)
def test_vector_add_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    vector_add(m, v)


def test_vector_sub():
    """
    [a b]  - [c d]  = [a-c b-d]
    Matrix + Matrix = Matrix
    """
    assert vector_sub(v, w) == [1, 1, -4]
    assert vector_sub(w, v) == [-1, -1, 4]
    assert vector_sub(y, z) == y
    assert vector_sub(w, u) == vector_sub(z, vector_sub(u, w))


@raises(ShapeError)
def test_vector_sub_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    vector_sub(m, v)


def test_vector_sum():
    """vector_sum can take any number of vectors and add them together."""
    assert vector_sum(v, w, u, y, z) == [12, 26, 35]


@raises(ShapeError)
def test_vector_sum_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    vector_sum(v, w, m, y)


def test_dot():
    """
    dot([a b], [c d])   = a * c + b * d
    dot(Vector, Vector) = Scalar
    """
    assert dot(w, y) == 160
    assert dot(m, n) == 15
    assert dot(u, z) == 0


@raises(ShapeError)
def test_dot_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    dot(v, m)


def test_vector_multiply():
    """
    [a b]  *  Z     = [a*Z b*Z]
    Vector * Scalar = Vector
    """
    assert vector_multiply(v, 0.5) == [0.5, 1.5, 0]
    assert vector_multiply(m, 2) == [6, 8]


def test_vector_mean():
    """
    mean([a b], [c d]) = [mean(a, c) mean(b, d)]
    mean(Vector)       = Vector
    """
    assert vector_mean(m, n) == [4, 2]
    assert vector_mean(v, w) == [0.5, 2.5, 2]
    assert are_equal(vector_mean(v, w, u)[0], 2 / 3)
    assert are_equal(vector_mean(v, w, u)[1], 2)
    assert are_equal(vector_mean(v, w, u)[2], 5 / 3)

@raises(ShapeError)
def test_vector_mean_checks_shapes():
    vector_mean(m,v)

def test_magnitude():
    """
    magnitude([a b])  = sqrt(a^2 + b^2)
    magnitude(Vector) = Scalar
    """
    assert magnitude(m) == 5
    assert magnitude(v) == math.sqrt(10)
    assert magnitude(y) == math.sqrt(1400)
    assert magnitude(z) == 0


A = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
B = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
C = [[1, 2],
     [2, 1],
     [1, 2]]
D = [[1, 2, 3],
     [3, 2, 1]]


# ADVANCED MODE TESTS BELOW
# UNCOMMENT THEM FOR ADVANCED MODE!

def test_shape_matrices():
    """shape takes a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    assert shape(A) == (3, 3)
    assert shape(C) == (3, 2)
    assert shape(D) == (2, 3)


def test_matrix_row():
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    assert matrix_row(A, 0) == [1, 0, 0]
    assert matrix_row(B, 1) == [4, 5, 6]
    assert matrix_row(C, 2) == [1, 2]


def test_matrix_col():
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    assert matrix_col(A, 0) == [1, 0, 0]
    assert matrix_col(B, 1) == [2, 5, 8]
    assert matrix_col(D, 2) == [3, 1]


def test_matrix_matrix_add():
    assert matrix_add(A, B) == [[2, 2, 3],
                                [4, 6, 6],
                                [7, 8, 10]]


@raises(ShapeError)
def test_matrix_add_checks_shapes():
    """Shape rule: the rows and columns of the matrices must be the same size."""
    matrix_add(C, D)


def test_matrix_matrix_sub():
    assert matrix_sub(A, B) == [[ 0, -2, -3],
                                [-4, -4, -6],
                                [-7, -8, -8]]


@raises(ShapeError)
def test_matrix_sub_checks_shapes():
    """Shape rule: the rows and columns of the matrices must be the same size."""
    matrix_sub(C, D)


def test_matrix_scalar_multiply():
    """
    [[a b]   *  Z   =   [[a*Z b*Z]
     [c d]]              [c*Z d*Z]]

    Matrix * Scalar = Matrix
    """
    assert matrix_scalar_multiply(C, 3) == [[3, 6],
                                            [6, 3],
                                            [3, 6]]
    assert matrix_scalar_multiply(B, 2) == [[ 2,  4,  6],
                                            [ 8, 10, 12],
                                            [14, 16, 18]]


def test_matrix_vector_multiply():
    """
    [[a b]   *  [x   =   [a*x+b*y
     [c d]       y]       c*x+d*y
     [e f]                e*x+f*y]

    Matrix * Vector = Vector
    """
    assert matrix_vector_multiply(A, [2, 5, 4]) == [2, 5, 4]
    assert matrix_vector_multiply(B, [1, 2, 3]) == [14, 32, 50]
    assert matrix_vector_multiply(C, [3, 4]) == [11, 10, 11]
    assert matrix_vector_multiply(D, [0, 1, 2]) == [8, 4]


@raises(ShapeError)
def test_matrix_vector_multiply_checks_shapes():
    """Shape Rule: The number of rows of the vector must equal the number of
    columns of the matrix."""
    matrix_vector_multiply(C, [1, 2, 3])


def test_matrix_matrix_multiply():
    """
    [[a b]   *  [[w x]   =   [[a*w+b*y a*x+b*z]
     [c d]       [y z]]       [c*w+d*y c*x+d*z]
     [e f]                    [e*w+f*y e*x+f*z]]

    Matrix * Matrix = Matrix
    """
    assert matrix_matrix_multiply(A, B) == B
    assert matrix_matrix_multiply(B, C) == [[8, 10],
                                            [20, 25],
                                            [32, 40]]
    assert matrix_matrix_multiply(C, D) == [[7, 6, 5],
                                            [5, 6, 7],
                                            [7, 6, 5]]
    assert matrix_matrix_multiply(D, C) == [[8, 10], [8, 10]]


@raises(ShapeError)
def test_matrix_matrix_multiply_checks_shapes():
    """Shape Rule: The number of columns of the first matrix must equal the
    number of rows of the second matrix."""
    matrix_matrix_multiply(A, D)

# ## End of code provided from:
# ## https://github.com/tiyd-python-2016-02/assignments/blob/master/week2/linear_algebra_test.py
#Tests for class versions of matrix and vector code

def test_Vector2d_constructor():
    assert Vector2d([1,2,3])._value == [1,2,3]

vector_m = Vector2d(m)
vector_n = Vector2d(n)

vector_u = Vector2d(u)
vector_v = Vector2d(v)
vector_w = Vector2d(w)
vector_y = Vector2d(y)
vector_z = Vector2d(z)

def test_shape_Vector2ds():
    print(shape([1]), Vector2d([]).shape())
    assert shape([1]) == Vector2d([1]).shape()
    assert shape(m) == vector_m.shape()
    assert shape(v) == vector_v.shape()

def test_Vector2d_add():
    """
    [a b]  + [c d]  = [a+c b+d]
    Matrix + Matrix = Matrix
    """
    assert (vector_v + vector_w)._value == vector_add(v, w) #== [1, 5, 4]
    assert (vector_u + vector_y)._value == vector_add(u, y) #== [11, 21, 31]
    assert (vector_u + vector_z)._value == vector_add(u, z) #== u

def test_Vector2d_eq():
    assert vector_u == Vector2d(u)
    assert vector_u != Vector2d(v)

@raises (ShapeError)
def test_Vector2d_eq_checks_shapes():
    Vector2d([1,1]) == Vector2d([1,1,1])

def test_Vector2d_add_is_commutative():
    assert vector_w + vector_y == vector_y + vector_w

@raises(ShapeError)
def test_Vector2d_add_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    vector_m + vector_v

def test_Vector2d_sub():
    """
    [a b]  - [c d]  = [a-c b-d]
    Matrix + Matrix = Matrix
    """
    assert (vector_v - vector_w)._value == vector_sub(v, w) #== [1, 1, -4]
    assert (vector_w - vector_v)._value == vector_sub(w, v) #== [-1, -1, 4]
    assert (vector_y - vector_z)._value == vector_sub(y, z) #== y
    assert (vector_w - vector_u)._value == vector_sub(w, u) #== vector_sub(z, vector_sub(u, w))
    assert vector_w - vector_u == vector_z - (vector_u - vector_w) #vector_sub(z, vector_sub(u, w))

@raises(ShapeError)
def test_Vector2d_sub_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    vector_m - vector_v

def test_Vector2d_sum():
    """vector_sum can take any number of vectors and add them together."""
    assert sum([vector_v, vector_w, vector_u, vector_y, vector_z])._value == vector_sum(v, w, u, y, z) #== [12, 26, 35]


@raises(ShapeError)
def test_Vector2d_sum_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    sum([vector_v, vector_w, vector_m, vector_y])


def test_Vector2d_dot():
    """
    dot([a b], [c d])   = a * c + b * d
    dot(Vector, Vector) = Scalar
    """
    assert vector_w.dot(vector_y) == dot(w, y) #== 160
    assert vector_m.dot(vector_n) == dot(m, n) #== 15
    assert vector_u.dot(vector_z) == dot(u, z) #== 0


@raises(ShapeError)
def test_Vector2d_dot_checks_shapes():
    """Shape rule: the vectors must be the same size."""
    vector_v.dot(vector_m)

def test_Vector2d_multiply():
    """
    [a b]  *  Z     = [a*Z b*Z]
    Vector * Scalar = Vector
    """
    assert (vector_v * 0.5)._value == vector_multiply(v, 0.5) #== [0.5, 1.5, 0]
    assert (vector_m * 2)._value == vector_multiply(m, 2) #== [6, 8]


def test_Vector2d_mean():
    """
    mean([a b], [c d]) = [mean(a, c) mean(b, d)]
    mean(Vector)       = Vector
    """
    assert Vector2d.mean(vector_m, vector_n)._value == vector_mean(m, n)# == [4, 2]
    assert Vector2d.mean(vector_v,vector_w)._value == vector_mean(v, w)# == [0.5, 2.5, 2]
    assert are_equal(Vector2d.mean(vector_v,vector_w, vector_u)._value[0],vector_mean(v, w, u)[0])#, 2 / 3)
    assert are_equal(Vector2d.mean(vector_v,vector_w, vector_u)._value[1],vector_mean(v, w, u)[1])#, 2)
    assert are_equal(Vector2d.mean(vector_v,vector_w, vector_u)._value[2],vector_mean(v, w, u)[2])#, 5 / 3)

@raises(ShapeError)
def test_Vector2d_mean_checks_shapes():
    Vector2d.mean(vector_m, vector_v)

def test_Vector2d_magnitude():
    """
    magnitude([a b])  = sqrt(a^2 + b^2)
    magnitude(Vector) = Scalar
    """
    assert vector_m.magnitude() == magnitude(m)# == 5
    assert vector_v.magnitude() == magnitude(v)# == math.sqrt(10)
    assert vector_y.magnitude() == magnitude(y)# == math.sqrt(1400)
    assert vector_z.magnitude() == magnitude(z)# == 0

def test_Matrix_constructor():
    assert Matrix(A)._value == A

matrix_A = Matrix(A)
matrix_B = Matrix(B)
matrix_C = Matrix(C)
matrix_D = Matrix(D)

def test_shape_Matrices():
    """shape takes a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    assert matrix_A.shape() == shape(A)# == (3, 3)
    assert matrix_C.shape() == shape(C)# == (3, 2)
    assert matrix_D.shape() == shape(D)# == (2, 3)


def test_Matrix_row():
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    assert matrix_A.row(0) == matrix_row(A, 0)# == [1, 0, 0]
    assert matrix_B.row(1) == matrix_row(B, 1)# == [4, 5, 6]
    assert matrix_C.row(2) == matrix_row(C, 2)# == [1, 2]

# def test_Matrix_get_transpose():
#     print(matrix_A.get_transpose()._value)
#     assert matrix_A.get_transpose() == matrix_A
#     assert matrix_B.get_transpose()._value == [[1,4,7],
#          [2,5,8],
#          [3,6,9]]
#
# def test_Matrix_col():
#     """
#            0 1  <- rows
#        0 [[a b]]
#        1 [[c d]]
#        ^
#      columns
#     """
#     assert matrix_A.col(0) == matrix_col(A, 0)# == [1, 0, 0]
#     assert matrix_B.col(1) == matrix_col(B, 1)# == [2, 5, 8]
#     assert matrix_C.col(2) == matrix_col(D, 2)# == [3, 1]


# def test_matrix_matrix_add():
#     assert matrix_add(A, B) == [[2, 2, 3],
#                                 [4, 6, 6],
#                                 [7, 8, 10]]
#
#
# @raises(ShapeError)
# def test_matrix_add_checks_shapes():
#     """Shape rule: the rows and columns of the matrices must be the same size."""
#     matrix_add(C, D)
#
#
# def test_matrix_matrix_sub():
#     assert matrix_sub(A, B) == [[ 0, -2, -3],
#                                 [-4, -4, -6],
#                                 [-7, -8, -8]]
#
#
# @raises(ShapeError)
# def test_matrix_sub_checks_shapes():
#     """Shape rule: the rows and columns of the matrices must be the same size."""
#     matrix_sub(C, D)
#
#
# def test_matrix_scalar_multiply():
#     """
#     [[a b]   *  Z   =   [[a*Z b*Z]
#      [c d]]              [c*Z d*Z]]
#
#     Matrix * Scalar = Matrix
#     """
#     assert matrix_scalar_multiply(C, 3) == [[3, 6],
#                                             [6, 3],
#                                             [3, 6]]
#     assert matrix_scalar_multiply(B, 2) == [[ 2,  4,  6],
#                                             [ 8, 10, 12],
#                                             [14, 16, 18]]
#
#
# def test_matrix_vector_multiply():
#     """
#     [[a b]   *  [x   =   [a*x+b*y
#      [c d]       y]       c*x+d*y
#      [e f]                e*x+f*y]
#
#     Matrix * Vector = Vector
#     """
#     assert matrix_vector_multiply(A, [2, 5, 4]) == [2, 5, 4]
#     assert matrix_vector_multiply(B, [1, 2, 3]) == [14, 32, 50]
#     assert matrix_vector_multiply(C, [3, 4]) == [11, 10, 11]
#     assert matrix_vector_multiply(D, [0, 1, 2]) == [8, 4]
#
#
# @raises(ShapeError)
# def test_matrix_vector_multiply_checks_shapes():
#     """Shape Rule: The number of rows of the vector must equal the number of
#     columns of the matrix."""
#     matrix_vector_multiply(C, [1, 2, 3])
#
#
# def test_matrix_matrix_multiply():
#     """
#     [[a b]   *  [[w x]   =   [[a*w+b*y a*x+b*z]
#      [c d]       [y z]]       [c*w+d*y c*x+d*z]
#      [e f]                    [e*w+f*y e*x+f*z]]
#
#     Matrix * Matrix = Matrix
#     """
#     assert matrix_matrix_multiply(A, B) == B
#     assert matrix_matrix_multiply(B, C) == [[8, 10],
#                                             [20, 25],
#                                             [32, 40]]
#     assert matrix_matrix_multiply(C, D) == [[7, 6, 5],
#                                             [5, 6, 7],
#                                             [7, 6, 5]]
#     assert matrix_matrix_multiply(D, C) == [[8, 10], [8, 10]]
#
#
# @raises(ShapeError)
# def test_matrix_matrix_multiply_checks_shapes():
#     """Shape Rule: The number of columns of the first matrix must equal the
#     number of rows of the second matrix."""
#     matrix_matrix_multiply(A, D)
