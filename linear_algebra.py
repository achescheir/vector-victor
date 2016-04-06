
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
    return [ sum(x) for x in zip(*args)]
