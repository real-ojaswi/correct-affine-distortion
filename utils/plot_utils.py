import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def draw_lines(points, img, title, output_dir):
    """
    Draws lines between given points on an image using OpenCV.

    Parameters:
    points (ndarray): Array of point pairs to be connected by lines (N x 2 x 2).
    img (ndarray): Image on which lines will be drawn.
    title (str): Title for saving the image.
    """
    # Create a copy of the image to avoid modifying the original
    img_copy = img.copy()

    # Draw lines between the given pairs of points
    for point_pair in points:
        start_point = (int(point_pair[0][0]), int(point_pair[0][1]))
        end_point = (int(point_pair[1][0]), int(point_pair[1][1]))

        # Draw the line on the image
        cv2.line(img_copy, start_point, end_point, (255, 0, 0), 10)  # Blue color line with thickness 10

    
    # Save the image with the lines drawn
    path= os.path.join(output_dir, f'{title}_lines.jpg')
    cv2.imwrite(path, img_copy)


def draw_points(points, img, title, output_dir):
    """
    Draws points on an image and annotates them with alphabets.

    Parameters:
    points (ndarray): Array of points to be drawn (N x 2).
    img (ndarray): Image on which points will be drawn.
    title (str): Title for saving the image.
    """
    # Copy image to avoid modifying the original
    annotated_img = img.copy()

    # Loop through each point and draw it with an annotation
    for i, point in enumerate(points):
        # Draw the point as a filled circle
        cv2.circle(annotated_img, (int(point[0]), int(point[1])), radius=5, color=(0, 0, 255), thickness=5)

        # Annotate the point with an alphabet
        text = chr(65 + i)  # Convert index to corresponding alphabet (A, B, C, ...)
        cv2.putText(annotated_img, text, (int(point[0]) + 10, int(point[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    # Save the image with points drawn and annotated
    path= os.path.join(output_dir, f'{title}_points.jpg')
    cv2.imwrite(path, annotated_img)
