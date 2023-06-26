import numpy as np
import cv2 as cv


def cross_product_2d(vector1, vector2):
    """
    takes two vectors as tuples, and returns their 2d cross product.
    """
    return vector1[0] * vector2[1] - vector1[1] * vector2[0]


def argmax_2d(matrix):
    """
    returns indexes of the cell in a matrix where it has the maximum value
    """
    max_index = np.argmax(matrix)
    index_2d = np.unravel_index(max_index, matrix.shape)
    return index_2d


def tuple_to_vector(vertices):
    """
    takes a list of two tuples, and returns the vector connecting the first to the second tuple.
    """
    a, b = vertices
    return [b[0] - a[0], b[1] - a[1]]


def draw(image, corners):
    """
    draw a polygon with the given corners on the input image
    :param image: the input image to draw the polygon on
    :param corners: the corners of the polygon
    :return:
    """
    corners = swap_touples(corners)
    corners = np.array(corners, dtype=np.int32)
    corners = corners.reshape((-1, 1, 2))
    colored_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    cv.polylines(colored_image, [corners], isClosed=True, color=(0, 0, 255), thickness=1)
    cv.imwrite("polygon.png", colored_image)


def swap_touples(lst):
    """
    takes a list of tuples and returns a list where the items in every tuple is swapped.
    """
    return [(t[1], t[0]) for t in lst]
