
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
    return list(list(zip(*matrix))[index])

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
