
class ShapeError(Exception):
    pass

def shape(matrix):
    # columns = [len(x), for x in matrix]
    # columns_length = min(columns):
    #
    # if columns is None:
        return (len(matrix),)
    # else
    #     return

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
