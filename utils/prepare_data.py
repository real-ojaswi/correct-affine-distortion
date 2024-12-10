import numpy as np
import cv2

from utils.utils import get_intersection, get_lines


def return_data(type):
    """
    Retrieves data specific to the provided image type, including key points, line equations, intersections, 
    and additional geometric details for distortion correction and geometric analysis.

    Parameters:
    type (str): The type of image to retrieve data for. Accepted values are 'board', 'corridor', 'tv', and 'wall'.

    Returns:
    tuple: A tuple containing the following:
        - image (numpy.ndarray): The loaded image corresponding to the specified type.
        - point_image (numpy.ndarray): A 2D array of key points in the image coordinates.
        - point_real (numpy.ndarray): A 2D array of corresponding key points in real-world coordinates.
        - intersection_1 (numpy.ndarray): Intersection of two parallel (in real world) lines in homogeneous coordinates.
        - intersection_2 (numpy.ndarray): Another intersection of two parallel (in real world) lines in homogeneous coordinates.
        - lines (list): A list of homogeneous coordinates of lines that are arranged such that every pair is orthogonal in world 2D space.
        - extra_lines (list): A list of additional homogeneous coordinates of lines.
        - line_points (list): A list of points used to define all the lines.
    """
    if type == 'board':
        point_image= np.array([[70, 421], [1222, 138], [1354, 1954], [420, 1791]])
        point_real= np.array([[300, 300], [800+300, 300], [800+300, 1200+300], [0+300, 1200+300]])
        line1_1_points= np.array([[421, 1790], [70, 422]]) # vertical left
        line1_2_points= np.array([[1354, 1955], [1222, 139]]) # vertical right
        line2_1_points= np.array([[70, 422], [1222, 138]]) # horizontal up
        line2_2_points= np.array([[421, 1790], [1354, 1955]]) # horizontal down
        image= cv2.imread('images/board_1.jpeg')

        line1_1= get_lines(line1_1_points[0], line1_1_points[1])
        line1_2= get_lines(line1_2_points[0], line1_2_points[1])

        line2_1= get_lines(line2_1_points[0], line2_1_points[1])
        line2_2= get_lines(line2_2_points[0], line2_2_points[1])

        intersection_1= get_intersection(line1_1, line1_2)
        intersection_2= get_intersection(line2_1, line2_2)

        intersection_1= intersection_1/intersection_1[2]
        intersection_2= intersection_2/intersection_2[2]

        # Additional lines
        line3_1_points= np.array([[1439, 2013], [1427, 564]]) # vertical rack
        line3_2_points= np.array([[1427, 564], [1510, 553]]) # horizontal rack

        line3_1= get_lines(line3_1_points[0], line3_1_points[1])
        line3_2= get_lines(line3_2_points[0], line3_2_points[1])


    if type == 'corridor':
        point_image= np.array([[1084, 529], [1305, 487], [1296, 1340], [1078, 1216]])
        point_real= np.array([[600, 600], [300+600, 600], [300+600, 600+600], [0+600, 600+600]])
        image= cv2.imread('images/corridor.jpeg')

        line1_1_points= np.array([[1084, 529], [1305, 487]]) # horizontal top
        line1_2_points= np.array([[1296, 1340], [1078, 1216]]) # horizontal down
        line2_1_points= np.array([[1084, 529], [1078, 1216]]) # vertical left
        line2_2_points= np.array([[1305, 487], [1296, 1340]]) # horizontal right

        line1_1= get_lines(line1_1_points[0], line1_1_points[1])
        line1_2= get_lines(line1_2_points[0], line1_2_points[1])

        line2_1= get_lines(line2_1_points[0], line2_1_points[1])
        line2_2= get_lines(line2_2_points[0], line2_2_points[1])

        intersection_1= get_intersection(line1_1, line1_2)
        intersection_2= get_intersection(line2_1, line2_2)

        intersection_1= intersection_1/intersection_1[2]
        intersection_2= intersection_2/intersection_2[2]

        # Additional lines
        line3_1_points= np.array([[815, 576], [920, 558]]) # vertical rack
        line3_2_points= np.array([[815, 576], [811, 1061]]) # horizontal rack

        line3_1= get_lines(line3_1_points[0], line3_1_points[1])
        line3_2= get_lines(line3_2_points[0], line3_2_points[1])


    if type == 'tv':
        point_image= np.array([[688, 1521], [2645,1122], [2785,3060], [748, 2824]])
        point_real= np.array([[600, 600], [730+600, 600], [730+600, 430+600], [0+600, 430+600]])
        image= cv2.imread('images/tv.jpg')

        line1_1_points= np.array([[688, 1521], [2645,1122]]) # horizontal top
        line1_2_points= np.array([[2785,3060], [748, 2824]]) # horizontal down
        line2_1_points= np.array([[688, 1521], [748, 2824]]) # vertical left
        line2_2_points= np.array([[2645,1122], [2785,3060]]) # vertical right

        line1_1= get_lines(line1_1_points[0], line1_1_points[1])
        line1_2= get_lines(line1_2_points[0], line1_2_points[1])

        line2_1= get_lines(line2_1_points[0], line2_1_points[1])
        line2_2= get_lines(line2_2_points[0], line2_2_points[1])

        intersection_1= get_intersection(line1_1, line1_2)
        intersection_2= get_intersection(line2_1, line2_2)

        intersection_1= intersection_1/intersection_1[2]
        intersection_2= intersection_2/intersection_2[2]

        # Additional lines
        line3_1_points= np.array([[429, 3259], [1805, 3582]]) # vertical rack
        line3_2_points= np.array([[429, 3259], [479, 3925]]) # horizontal rack

        line3_1= get_lines(line3_1_points[0], line3_1_points[1])
        line3_2= get_lines(line3_2_points[0], line3_2_points[1])


    if type == 'wall':
        point_image= np.array([[2155, 2735], [2619,2750], [2733,3912], [2221, 3926]])
        point_real= np.array([[600, 600], [710+600, 600], [710+600, 330+600], [0+600, 330+600]])
        image= cv2.imread('images/wall.jpg')

        line1_1_points= np.array([[2155, 2735], [2619,2750]]) # horizontal top
        line1_2_points= np.array([[2733,3912], [2221, 3926]]) # horizontal down
        line2_1_points= np.array([[2155, 2735], [2221, 3926]]) # vertical left
        line2_2_points= np.array([[2619,2750], [2733,3912]]) # vertical right

        line1_1= get_lines(line1_1_points[0], line1_1_points[1])
        line1_2= get_lines(line1_2_points[0], line1_2_points[1])

        line2_1= get_lines(line2_1_points[0], line2_1_points[1])
        line2_2= get_lines(line2_2_points[0], line2_2_points[1])

        intersection_1= get_intersection(line1_1, line1_2)
        intersection_2= get_intersection(line2_1, line2_2)

        intersection_1= intersection_1/intersection_1[2]
        intersection_2= intersection_2/intersection_2[2]

        # Additional lines
        line3_1_points= np.array([[2033, 2169], [2964, 2243]]) # vertical wall edge
        line3_2_points= np.array([[2033, 2169], [2055, 2633]]) # horizontal wall edge

        line3_1= get_lines(line3_1_points[0], line3_1_points[1])
        line3_2= get_lines(line3_2_points[0], line3_2_points[1])

    lines = [line1_1, line2_1, line1_2, line2_2, line1_1, line2_2, line1_2, line2_1] # to be arranged such that every pair of lines is orthogonal in world 2d
    extra_lines= [line3_1, line3_2]
    line_points= [point for point in [line for line in [line1_1_points, line1_2_points, line2_1_points, line2_2_points, line3_1_points, line3_2_points]]]

    return image, point_image, point_real, intersection_1, intersection_2, lines, extra_lines, line_points 