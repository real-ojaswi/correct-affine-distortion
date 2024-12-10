import numpy as np

def get_lines(point1, point2):
    point1= np.array([point1[0], point1[1], 1])
    point2= np.array([point2[0], point2[1], 1])
    line= np.cross(point1, point2)
    return line


def get_intersection(line1, line2):
    intersection= np.cross(line1, line2)
    return intersection


def transform_lines(lines, H):
    lines_transformed= []
    for line in lines:
        line_trans= np.transpose(np.linalg.inv(H)) @ line
        line_trans /= np.linalg.norm(line_trans)
        lines_transformed.append(line_trans)
    return lines_transformed