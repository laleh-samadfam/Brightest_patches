# Press Shift+F10 to execute it or replace it with your code.
import cv2 as cv
import numpy as np
from utils import *
from argparse import ArgumentParser


# BRIGHTEST PATCH CALCULATION FUNCTIONS
def calc_integral(inp):
    """
    calculate the integral image of the input image. each cell in the integral image is the accumulative intensity of
    the rectangle which its right bottom corner ends in that cell.
    :param inp: input image
    :return: integral image
    """
    integral_image = np.zeros((inp.shape[0], inp.shape[1]))
    integral_image[0, 0] = inp[0, 0]
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            if i > 0 and j > 0:
                integral_image[i, j] = inp[i, j] + integral_image[i, j - 1] + integral_image[i - 1, j] - integral_image[
                    i - 1, j - 1]
            elif i > 0 and j == 0:
                integral_image[i, j] = inp[i, j] + integral_image[i - 1, j]
            elif i == 0 and j > 0:
                integral_image[i, j] = inp[i, j] + integral_image[i, j - 1]
    return integral_image


def calc_patch_brightness(im, i, j, patch_size):
    """
    calculates the average brightness of a patch of an image marked by the right bottom corner cell
     and the size of the patch
    :param im: integral image of the image
    :param i: row number of right bottom corner of the patch
    :param j: column number of the right bottom corner of the patch
    :param patch_size: size of patch in form of (k, l)
    :return: average brightness of the patch
    """

    if i == (patch_size[0] - 1) and j == (patch_size[1] - 1):
        sum_brightness = im[i, j] / (patch_size[0] * patch_size[1])
    elif i == (patch_size[0] - 1) and j > (patch_size[1] - 1):
        sum_brightness = im[i, j] - im[i, j - patch_size[1]]
    elif i > (patch_size[0] - 1) and j == (patch_size[1] - 1):
        sum_brightness = im[i, j] - im[i - patch_size[0], j]
    else:
        sum_brightness = im[i, j] + im[i - patch_size[0], j - patch_size[1]] - im[i - patch_size[0], j] - im[
            i, j - patch_size[1]]
    return sum_brightness / (patch_size[0] * patch_size[1])


def calc_brightness_image(integral_image, patch_size):
    """
    Calculates an image in which the value of cell (i,j) is the average brightness of the patch_size patch around that
    cell in the input image using its integral image.
    :param integral_image: Integral image of the input image
    :param patch_size: size of patches that the average is calculated through it.
    :return: an image which each cell shows the average brightness of a patch of a given size around that cell
    """
    row, col = integral_image.shape
    k, l = patch_size
    brightness_image = np.zeros((row, col))

    for i in range(k // 2, row - k // 2):
        for j in range(l // 2, col - l // 2):
            brightness_image[i, j] = calc_patch_brightness(integral_image, i + k // 2, j + l // 2, patch_size)

    return brightness_image


def find_max_patch(brightness_image):
    return argmax_2d(brightness_image)


def remove_patch(mask, patch_center, patch_size):
    """
    Removes a patch from the available pixels of the matrix
    :param mask: ndarray, indicating available pixels
    :param patch_center: touple, center of the patch to be removed from the mask
    :param patch_size: touple, size of the patch to be removed
    :return: the new mask, with the given patch removed from it.
    """
    i, j = patch_center
    k, l = patch_size
    m, n = mask.shape

    if i < k // 2 or j < l // 2 or i >= m - k // 2 or j >= n - l // 2:
        raise ValueError("Unacceptable value for the center of the patch.")

    top = max(0, i - k + 1)
    left = max(0, j - l + 1)
    right = min(n, j + l)
    bottom = min(m, i + k)

    mask[top:bottom, left:right] = 0
    return mask


def find_max_patches(im, num_patches, patch_size):
    """
    Finds the maximum brightness patches in an image.

    :param  im: ndarray, Input image.
    :param num_patches: int, Number of maximum patches to find.
    :param patch_size: tuple, Size of the patches.

    :returns list: List of tuples representing the coordinates of the maximum patches.
    """

    integral_image = calc_integral(im)
    brightness_image = calc_brightness_image(integral_image, patch_size)

    m, n = brightness_image.shape
    k, l = patch_size
    top = []
    mask = np.ones_like(brightness_image)
    mask[:, :] = 1
    for row in range(m):
        for col in range(n):
            if row < k // 2 or col < l // 2 or row >= m - k // 2 or col >= n - l // 2:
                mask[row][col] = 0
    for i in range(num_patches):
        brightness_image = np.multiply(brightness_image, mask)
        max_patch = find_max_patch(brightness_image)
        mask = remove_patch(mask, max_patch, patch_size)
        top.append(max_patch)
    return top


# FINDING QUADRILATERAL AND IT'S AREA FUNCTIONS

def is_line(edge_mat):
    return np.all(edge_mat == 0)


def is_triangle(cross_mat):
    zero_cells = np.where(cross_mat == 0)
    return len(zero_cells[0]) > 6


def calc_triangle_area(cross_matrix, index):
    return abs(cross_matrix[index] / 2)


def calc_triangle_ara(vertices):
    """
    Calculates the area of a given triangle
    :param vertices: vertices of the triangle
    :return: area
    """
    a, b, c = vertices
    ac = [c[0] - a[0], c[1] - a[1]]
    ab = [b[0] - a[0], b[1] - a[1]]
    cross = abs(cross_product_2d(ac, ab))

    return cross / 2


def calc_quadrilateral_area(vertices):
    """
    Calculates the area of a non-self-intersecting quadrilateral in which the edge connecting 0sth and 2nd vertices
    is inside the quadrilateral using triangulation.
    :param vertices: ordered vertices to form a non-self-intersecting quadrilateral
    :return: area of the quadrilaterael
    """
    a, b, c, d = vertices
    return calc_triangle_ara([a, c, b]) + calc_triangle_ara([a, c, d])


def is_opposite_side(edge, points):
    """
    Determins if two points are on the same or opposite sides of a given edge
    :param edge: list of two touples formatting the edge
    :param points: list of two touples
    :return: True if the points are on the opposite side, False if not
    """
    a, c = edge
    b, d = points
    ac = touple_to_vector([a, c])
    ab = touple_to_vector([a, b])
    ad = touple_to_vector([a, d])

    return (cross_product_2d(ac, ab) * cross_product_2d(ac, ad)) < 0


def is_not_self_intersecting(quad):
    return is_opposite_side([quad[0], quad[2]], [quad[1], quad[3]])


def find_line(corners):
    return 0, corners


def find_triangle(map, cross_matrix):
    """
    finds the correct order of the vertices that form the biggest triangle, and the area of the triangle
    :param map: dictionary, mapping the edges to indexes in the cross_matrix
    :param cross_matrix: ndarray,
    :return:
    """
    row, col = argmax_2d(cross_matrix)  # find the triangle with the biggest area

    area = calc_triangle_area(cross_matrix, (row, col))  # find area
    vertices= []
    vertices.append(map[row][0])
    vertices.append(map[row][1])
    vertices.append(map[col][1])

    #vertices = [map[row], map[col]]  # 3 vertices that span the biggest triangle
    #vertices = list(set(vertices))  # removing the redundant vertice
    return area, vertices


def find_quadrilateral(corners):
    """
    Returns a list of vertices that form a non-self-intersecting quadrilateral.
    :param corners: List, List of four corner vertices in the correct order
    :return: Tuple, containing the quadrilateral's area and its vertices
    """
    x, y, z, w = corners
    quadrilaterals = [
        corners,
        [x, z, y, w],
        [x, y, w, z]
    ]

    for quad in quadrilaterals:
        if is_not_self_intersecting(quad):
            return calc_quadrilateral_area(quad), quad

    raise Exception("The indices do not form a quadrilateral")


def fill_cross_mat(map):
    """
    fill in a matrix with cross products of every two possible combination of two edges derived from 4 corner points
    :param corners: list of touples,  showing the vertices
    :param map: dictoinary, mapping numbers to wanted edges.
    :return :ndarray, a matrix where the cross product of the given edges are stored at
    """
    cross_mat = np.zeros((6, 6))  # 6 candidate edges for the quadrilateral
    for i in range(cross_mat.shape[0]):
        for j in range(cross_mat.shape[1]):
            edge_i = touple_to_vector(map[i])
            edge_j = touple_to_vector(map[j])
            cross_mat[i, j] = cross_product_2d(edge_i, edge_j)
    return cross_mat


def find_area(corners):
    """
    Determines the shape and calculates the area of the quadrilateral formed by the given vertices.

    :param corners, list, List of vertices (not sorted) to form a quadrilateral.
    :returns tuple, A tuple containing the calculated area and the list of vertices in the correct order.
    """

    a, b, c, d = corners
    vertex_map = {
        0: [a, b], 1: [a, c], 2: [a, d], 3: [b, c], 4: [b, d], 5: [c, d]
    }

    cross_mat = fill_cross_mat(vertex_map)
    if is_line(cross_mat):
        return find_line(corners)
    elif is_triangle(cross_mat):
        return find_triangle(vertex_map, cross_mat)
    else:
        return find_quadrilateral(corners)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_image", default='', help="path to the input image")
    parser.add_argument("--patch_size", default=[5, 5], help="patch size")
    opt = parser.parse_args()

    # Read input image
    inp = cv.imread(opt.input_image, cv.IMREAD_GRAYSCALE)

    # Check if the image size is too small
    if inp.shape[0] < opt.patch_size[0] * 4 or inp.shape[1] < opt.patch_size[1] * 4:
        raise ValueError("Image size is too small.")

    # Find corners of maximum patches
    corners = find_max_patches(inp, 4, opt.patch_size)

    # Calculate area and order of vertices
    area, vertices = find_area(corners)
    print("The center of the brightest patches span an area of:", area, "in pixels scale.")

    # Draw the quadrilateral on the input image
    draw(inp, vertices)
