"""Question 1."""

"""
This requires implementing the marker-based watershed method -
Meyer’s Flooding or immersion method--as discussed in class.
Ready made function calls to watersheds should not be used
in the implementation. It is a good idea to first test the method on very
small images (say 4x4 or 8x8) to make sure that you dont have bugs.
Remember to compute the gradient image and perform the
flooding on the gradient image!."""


# !/usr/bin/python
import sys
from sets import Set
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from Queue import PriorityQueue
import numpy as np
from matplotlib import pyplot as plt

# Input is expected to be
# 1: path to a seed file
# 2: path to an image to segment
ARG_NR = 3

if len(sys.argv) < ARG_NR:
    raise RuntimeError("Missing input arguments")
else:
    seed_path = sys.argv[1]
    img_path = sys.argv[2]


def get_seed_data(seeds, seed_file):
    """Retrieve seed data from input text file."""
    for line in seed_file:
        line_elems = line.split(" ")
        line_elems = [elem for elem in line_elems if elem != '']
        seed_id = int(float(line_elems[0]))
        seed_col = int(float(line_elems[1]))
        seed_row = int(float(line_elems[2].split("\n")[0]))
        seeds.append((seed_id*20, seed_col, seed_row))


def imshow(img):
    """Display grayscale image."""
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram(img):
    """Compute pixel value histogram."""
    return cv2.calcHist([img], [0], None, [256], [0, 256])


def display_histogram(histogram):
    """Display histogram."""
    plt.hist(histogram, 256, [0, 256])
    plt.show()


def gradient_img(img):
    """Make gradient image from grayscale image."""
    # gradient = cv2.Sobel(img, cv2.CV_64F, 1, 1)
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient


def import_grayscale_img(img_path):
    """Import grayscale image at filepath."""
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def pixel_neighborhood(row, col, img):
    """Retrieve the neighboring pixels of pixel at location."""
    h, w = img.shape
    neighborhood = []
    cols = []
    rows = []

    # Check edge cases
    if col == 0:
        cols.append(col)
        cols.append(col + 1)
    elif col == w - 1:
        cols.append(col - 1)
        cols.append(col)
    else:
        cols.append(col - 1)
        cols.append(col)
        cols.append(col + 1)

    if row == 0:
        rows.append(row)
        rows.append(row + 1)
    elif row == h - 1:
        rows.append(row - 1)
        rows.append(row)
    else:
        rows.append(row + 1)
        rows.append(row)
        rows.append(row - 1)

    for px_col in cols:
        for px_row in rows:
            if not (px_col == col and px_row == row):
                intensity = img[px_row, px_col]
                neighborhood.append((intensity, px_row, px_col))

    return neighborhood


def neighbor_labels(row, col, label_img):
    """Extract the unique pixel values in an image neighborhood."""
    neighbors = pixel_neighborhood(row, col, label_img)
    neighbor_labels = Set()

    for neighbor in neighbors:
        neighbor_row = neighbor[1]
        neighbor_col = neighbor[2]
        # 300 means it is not labeled, 400 means it is a watershed line
        if label_img[neighbor_row, neighbor_col] != 300 and label_img[neighbor_row, neighbor_col] != 400:
            neighbor_labels.add(label_img[neighbor_row, neighbor_col])

    return neighbor_labels


def is_homogenous_neighborhood(row, col, label_img):
    """Check if all the labaled neighbors of a pixel have the same value."""
    neighbor_label_values = neighbor_labels(row, col, label_img)
    return (len(neighbor_label_values) < 2)


def meyers_flooding_algorithm(img, seeds):
    """Segment image based on Meyers flooding algorithm."""
    # Create a priority queue that can hold inf elements
    q = PriorityQueue(0)
    h, w = img.shape
    label_img = np.ones((h, w), dtype=np.uint8) * 300
    gradient = gradient_img(img)

    # The neighboring pixels of each seed pixel are
    # inserted into a priority queue with a priority level
    # corresponding to the inverse of the gradient magnitude of the pixel.
    for seed in seeds:
        seed_id = seed[0]
        col = seed[1]
        row = seed[2]
        gradient[row, col] = seed_id
        label_img[row, col] = seed_id
        neighborhood = pixel_neighborhood(row, col, gradient)
        for neighbor in neighborhood:
            if neighbor not in q.queue:
                q.put(neighbor)

    # The pixel with the lowest priority level is extracted from
    # the priority queue until the queue is empty
    while not q.empty():
        px = q.get()
        row = px[1]
        col = px[2]
        # If the neighbors of the extracted pixel that have already
        # been labeled all have the same label, then the pixel is
        # labeled with their label. All non-marked neighbors that
        # are not yet in the priority queue are put into the priority queue.
        neighbor_label_values = neighbor_labels(row, col, label_img)
        if len(neighbor_label_values) == 1:
            # It is not a wathershed line
            gradient[row, col] = neighbor_label_values.pop()
            label_img[row, col] = gradient[row, col]
            px_neighbors = pixel_neighborhood(row, col, gradient)
            for px_neighbor in px_neighbors:
                px_neighbor_row = px_neighbor[1]
                px_neighbor_col = px_neighbor[2]
                if label_img[px_neighbor_row, px_neighbor_col] == 300:
                    if px_neighbor not in q.queue:
                        q.put(px_neighbor)
        elif len(neighbor_label_values) > 1:
            # It is a wathershed line
            gradient[row, col] = 400
            label_img[row, col] = gradient[row, col]

    cv2.imwrite('gradient.jpg', gradient)



np.set_printoptions(threshold='nan')
seeds = []
seed_file = open(seed_path, 'r').readlines()
get_seed_data(seeds, seed_file)

img = import_grayscale_img(img_path)

test_row = [10, 15, 128, 128]
test_img = np.matrix([test_row, test_row, test_row, test_row], dtype=np.uint8)
test_seeds = [(1, 0, 0), (2, 3, 3)]

meyers_flooding_algorithm(img, seeds)
