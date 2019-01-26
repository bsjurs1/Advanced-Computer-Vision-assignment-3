"""Question 2."""

"""
Here you will implement the graph-cut method discussed in class.
Demonstrate the working of this method on the image beach.gif.
The implementation for efficient graph-based segmentation should
use the union-find algorithm (using the disjoint-set data structure).
Similar to Q1, you will demonstrate its working to me.
"""

# !/usr/bin/python
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np
import math
from Queue import PriorityQueue
from sklearn.neighbors import NearestNeighbors

# Utilization of PRIORITY QUEUES USING HEAP QUEUE
# http://docs.python.org/library/heapq.html is preferable
# HW 3 - Graph-Cut method
# Here you will implement the graph-cut method discussed in class.
# Demonstrate the working of this method
# on the image beach.gif and inputImage.jpg.
# The implementation for efficient graph-based
# segmentation should use the
# union-find algorithm (using the disjoint-set data structure).


# Arguments:
# inputImageFile - input image file
# threshold_K - threshold parameter for the graphcut algorithm
# outputImageFile - output image file in which segmentation result is stored

inf = float('inf')


def get_row_col(n, w):
    """Compute the row and col from element number n."""
    col = int(n % w)
    row = int(math.floor(float(n) / float(w)))
    return (row, col)


def get_elem_nr(row, col, w):
    """Compute the element nr from row and col."""
    return (row * w) + col


def eight_neighborhood(row, col, img):
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
                neighborhood.append((px_row, px_col))

    return neighborhood


def imshow(img):
    """Display grayscale image."""
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def import_grayscale_img(img_path):
    """Import grayscale image at filepath."""
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def tau(k, c):
    """Compute the weight to the int function."""
    if c == 0:
        return inf
    return float(k) / float(c)


def internal(ci):
    """Find the maximum edge weight of the MST for a certain component ci."""
    if len(ci.keys()) == 0:
        return inf
    else:
        max_e_weight = 0
        for row in ci.keys():
            for col in ci[row]:
                if ci[row][col] > max_e_weight:
                    max_e_weight = ci[row][col]
        return max_e_weight


def construct_adj_list(img):
    """Construct and adjacency list based on an image."""
    h, w = img.shape
    n = h * w
    adj_list = [{}] * n

    for row in range(n):
        im_row_i, im_col_i = get_row_col(row, w)
        px_i = img[im_row_i, im_col_i]
        neighbourhood = eight_neighborhood(im_row_i, im_col_i, img)
        neighbours = {}
        for neighbour_row, neighbour_col in neighbourhood:
            col = get_elem_nr(neighbour_row, neighbour_col, w)
            # diagonal should be 0 and avoid
            # adding a node multiple times by only keeping the upper part
            # of adjacency matrix
            if row == col:
                neighbours[col] = 0
            else:
                px_j = img[neighbour_row, neighbour_col]
                e_weight = abs(float(px_i) - float(px_j))
                neighbours[col] = e_weight
        adj_list[row] = neighbours

    return adj_list


def construct_init_segment_img(h, w):
    """Construct the initial segmented image."""
    segment_img = np.ones((h, w), dtype=np.int)
    segment_value = 0

    # Make each pixel its own component by giving everyone their own label
    for row in range(h):
        for col in range(w):
            segment_img[row, col] = segment_value
            segment_value += 1

    return segment_img


def get_edge_count(component):
    """Get the number of edges present in a component."""
    edge_count = 0
    for row in component.keys():
        edge_count += len(component[row].keys())
    return edge_count


def construct_component_msts(n, w, segment_img):
    """Construct the datastructure to hold each components MST."""
    component_msts = {}

    # Compute the init components mst
    for row in range(n):
        row_i, col_i = get_row_col(row, w)
        segment_label = segment_img[row_i, col_i]
        component_msts[segment_label] = {}
        component_msts[segment_label][row] = {}

    return component_msts


def mint(ci, cj, k):
    """Compute the MINT function."""
    ci_edge_count = get_edge_count(ci)
    cj_edge_count = get_edge_count(cj)
    return min(internal(ci) + tau(k, ci_edge_count),
               internal(cj) + tau(k, cj_edge_count))


def construct_priority_queue(img, n):
    """Construct a priority queue sorted on edge weights."""
    h, w = img.shape
    n = h * w
    adj_list = [{}] * n

    for row in range(n):
        im_row_i, im_col_i = get_row_col(row, w)
        px_i = img[im_row_i, im_col_i]
        neighbourhood = eight_neighborhood(im_row_i, im_col_i, img)
        neighbours = {}
        for neighbour_row, neighbour_col in neighbourhood:
            col = get_elem_nr(neighbour_row, neighbour_col, w)
            # diagonal should be 0 and avoid
            # adding a node multiple times by only keeping the upper part
            # of adjacency matrix
            if row == col:
                neighbours[col] = 0
            elif col >= row:
                continue
            else:
                px_j = img[neighbour_row, neighbour_col]
                e_weight = abs(float(px_i) - float(px_j))
                neighbours[col] = e_weight
        adj_list[row] = neighbours

    q = PriorityQueue(0)
    for row in range(n):
        for col in adj_list[row].keys():
            e_weight = adj_list[row][col]
            q.put((e_weight, row, col))
    return q


def kruskal(g):
    """Find MST of graph g with Kruskals algorithm."""
    e = PriorityQueue(0)
    A = {}
    nodesAdded = set()

    for row in g.keys():
        for col in g[row].keys():
            edge = (g[row][col], row, col)
            if col > row:
                A[row] = {}
                A[col] = {}
                e.put(edge)

    while not e.empty():
        edge = e.get()
        e_weight = edge[0]
        u = edge[1]
        v = edge[2]
        if u or v not in nodesAdded:
            A[u][v] = e_weight
            A[v][u] = e_weight
            nodesAdded.add(u)
            nodesAdded.add(v)

    return A



def graph_cut(in_img_path, threshold_k, out_img_path):
    """Segment image by using the graph-cut algorithm."""
    img = import_grayscale_img(in_img_path)
    #blur image slightly
    img = cv2.GaussianBlur(img, (5, 5), 0.8, 0.8)
    h, w = img.shape
    n = h * w

    segment_img = construct_init_segment_img(h, w)
    component_msts = construct_component_msts(n, w, segment_img)

    q = construct_priority_queue(img, n)
    m = len(q.queue)
    for i in range(m):
        print(i)
        e_weight, row, col = q.get()

        row_i, col_i = get_row_col(row, w)
        vi = segment_img[row_i, col_i]

        row_j, col_j = get_row_col(col, w)
        vj = segment_img[row_j, col_j]

        ci = component_msts[vi]
        cj = component_msts[vj]

        if vi != vj:
            if e_weight <= mint(ci, cj, threshold_k):
                if vi < vj:
                    # Put edge in ci
                    if row not in component_msts[vi].keys():
                        component_msts[vi][row] = {}
                    component_msts[vi][row][col] = e_weight

                    if col not in component_msts[vi].keys():
                        component_msts[vi][col] = {}
                    component_msts[vi][col][row] = e_weight

                    # Mark all vj segments as vi segments
                    for row_vj in component_msts[vj].keys():
                        row_row_vj, col_row_vj = get_row_col(row_vj, w)
                        segment_img[row_row_vj, col_row_vj] = vi
                        for col_vj in component_msts[vj][row_vj]:
                            row_col_vj, col_col_vj = get_row_col(col_vj, w)
                            segment_img[row_col_vj, col_col_vj] = vi

                    # Put all elements in vj to vi
                    for row_vj in component_msts[vj].keys():
                        if row_vj not in component_msts[vi].keys():
                            component_msts[vi][row_vj] = component_msts[vj][row_vj]
                        else:
                            for col_vj in component_msts[vj][row_vj]:
                                if col_vj not in component_msts[vi][row_vj]:
                                    component_msts[vi][row_vj][col_vj] = component_msts[vj][row_vj][col_vj]

                    # Find the MST for component ci and save it
                    ci = kruskal(ci)

                    # Delete vj
                    del component_msts[vj]

                elif vj < vi:
                    # Put edge in ci
                    if row not in component_msts[vj].keys():
                        component_msts[vj][row] = {}
                    component_msts[vj][row][col] = e_weight

                    if col not in component_msts[vj].keys():
                        component_msts[vj][col] = {}
                    component_msts[vj][col][row] = e_weight

                    # Mark all vi segments as vj segments
                    for row_vi in component_msts[vi].keys():
                        row_row_vi, col_row_vi = get_row_col(row_vi, w)
                        segment_img[row_row_vi, col_row_vi] = vj
                        for col_vi in component_msts[vi][row_vi]:
                            row_col_vi, col_col_vi = get_row_col(col_vi, w)
                            segment_img[row_col_vi, col_col_vi] = vj

                    # Put all elements in vj to vi
                    for row_vi in component_msts[vi].keys():
                        if row_vi not in component_msts[vj].keys():
                            component_msts[vj][row_vi] = component_msts[vi][row_vi]
                        else:
                            for col_vi in component_msts[vi][row_vi]:
                                if col_vi not in component_msts[vi][row_vi]:
                                    component_msts[vi][row_vi][col_vi] = component_msts[vi][row_vi][col_vi]
                    # Delete vj

                    # Find the MST for component cj and save it
                    cj = kruskal(cj)

                    del component_msts[vi]

    counter = 0
    mapping = {}
    for row in range(h):
        for col in range(w):
            label = segment_img[row, col]
            if label not in mapping.keys():
                mapping[label] = counter
                counter += 10
            segment_img[row, col] = mapping[label]

    imshow(segment_img)
    cv2.imwrite(out_img_path, segment_img)

def main():
    """Start exection of program."""
    if len(sys.argv) < 4:
        print 'Usage: graphcut.py inputImageFile threshold_K outputImageFile'
        sys.exit(1)

    in_img_path = sys.argv[1]
    threshold_k = sys.argv[2]
    out_img_path = sys.argv[3]

    graph_cut(in_img_path, threshold_k, out_img_path)


if __name__ == '__main__':
    main()
