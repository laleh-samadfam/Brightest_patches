from argparse import ArgumentParser
from brightest_quadrilateral import *

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
