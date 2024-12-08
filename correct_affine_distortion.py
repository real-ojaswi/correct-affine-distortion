#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[ ]:


def compute_homography(src_pts, dst_pts):
    """
    Computes the homography matrix using Direct Linear Transformation (DLT).
    
    Parameters:
    src_pts (ndarray): Source points in the first image (Nx2).
    dst_pts (ndarray): Corresponding destination points in the second image (Nx2).
    
    Returns:
    H (ndarray): The 3x3 homography matrix.
    """
    num_points = src_pts.shape[0]
    A = []

    # Construct the matrix A from the point correspondences
    for i in range(num_points):
        x_src, y_src = src_pts[i]
        x_dst, y_dst = dst_pts[i]
        A.append([-x_src, -y_src, -1, 0, 0, 0, x_dst * x_src, x_dst * y_src, x_dst])
        A.append([0, 0, 0, -x_src, -y_src, -1, y_dst * x_src, y_dst * y_src, y_dst])

    A = np.array(A)

    # Solve for the homography matrix using SVD
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))

    # Normalize the homography matrix so that H[2, 2] = 1
    H /= H[2, 2]

    return H


# In[ ]:


def transform_point(H, point):
    """Transforms a point using a homography matrix H."""
    homogeneous_point = np.array([point[0], point[1], 1])
    transformed_point = H @ homogeneous_point
    x_new = int(transformed_point[0] / transformed_point[2])
    y_new = int(transformed_point[1] / transformed_point[2])
    return np.array([x_new, y_new], dtype=int)

def compute_corner_points(H, image):
    """Computes the transformed corner points of an image using homography."""
    h, w = image.shape[:2]
    corners = np.zeros((4, 2), dtype=int)
    corners[0] = transform_point(H, (0, 0))
    corners[1] = transform_point(H, (w, 0))
    corners[2] = transform_point(H, (w, h))
    corners[3] = transform_point(H, (0, h))
    return corners

def calculate_output_dimensions(corner_points):
    """Calculates the size of the output image and the required translation offsets."""
    min_coords = np.min(corner_points, axis=0)
    max_coords = np.max(corner_points, axis=0)

    # Output image dimensions
    output_width = max_coords[0] - min_coords[0] + 1
    output_height = max_coords[1] - min_coords[1] + 1
    offset_x, offset_y = min_coords
    print(output_width, output_height, offset_x, offset_y)
    return output_width, output_height, offset_x, offset_y




# In[ ]:


def apply_homography(src_img, H):
    """
    Applies a homography transformation to an image.
    
    Parameters:
    src_img (ndarray): Source image to be warped.
    H (ndarray): The 3x3 homography matrix.
    
    Returns:
    warped_img (ndarray): The warped image after applying the homography.
    """
    # Initialize the output image with zeros
    corner_points= compute_corner_points(H, src_img)
    width, height, offset_x, offset_y= calculate_output_dimensions(corner_points)
    
    warped_img = np.zeros((height, width, src_img.shape[2]), dtype=src_img.dtype)
    
    # Compute the inverse of the homography matrix for backward mapping
    H_inv = np.linalg.inv(H)

    # Iterate over every pixel in the output image
    for y in range(height):
        for x in range(width):
            # Create homogeneous coordinate for the current pixel
            dest_coord = np.array([x+offset_x, y+offset_y, 1])
            
            # Map the pixel from the output image back to the source image using inverse homography
            src_coord = H_inv @ dest_coord
            src_coord /= src_coord[2]  # Convert to Cartesian coordinates

            x_src, y_src = src_coord[0], src_coord[1]

            # Check if the mapped source coordinates are within the valid range of the source image
            if 0 <= x_src < src_img.shape[1] and 0 <= y_src < src_img.shape[0]:
                # Perform bilinear interpolation to compute the pixel value
                x1, y1 = int(x_src), int(y_src)
                x2, y2 = min(x1 + 1, src_img.shape[1] - 1), min(y1 + 1, src_img.shape[0] - 1)

                a = x_src - x1
                b = y_src - y1

                # Calculate the interpolated pixel value
                interpolated_value = (
                    (1 - a) * (1 - b) * src_img[y1, x1] +
                    a * (1 - b) * src_img[y1, x2] +
                    (1 - a) * b * src_img[y2, x1] +
                    a * b * src_img[y2, x2]
                )

                # Assign the computed pixel value to the output image
                warped_img[y, x] = interpolated_value

    return warped_img


# In[ ]:


# #  Only for corridor image
# def apply_homography(src_img, H):
#     """
#     Applies a homography transformation to an image.
    
#     Parameters:
#     src_img (ndarray): Source image to be warped.
#     H (ndarray): The 3x3 homography matrix.
    
#     Returns:
#     warped_img (ndarray): The warped image after applying the homography.
#     """
#     # Initialize the output image with zeros
#     corner_points= compute_corner_points(H, src_img)
#     width, height, offset_x, offset_y= (6000, 6000, -6000, -4500)
#     warped_img = np.zeros((height, width, src_img.shape[2]), dtype=src_img.dtype)
    
#     # Compute the inverse of the homography matrix for backward mapping
#     H_inv = np.linalg.inv(H)

#     # Iterate over every pixel in the output image
#     for y in range(height):
#         for x in range(width):
#             # Create homogeneous coordinate for the current pixel
#             dest_coord = np.array([x+offset_x, y+offset_y, 1])
            
#             # Map the pixel from the output image back to the source image using inverse homography
#             src_coord = H_inv @ dest_coord
#             src_coord /= src_coord[2]  # Convert to Cartesian coordinates

#             x_src, y_src = src_coord[0], src_coord[1]

#             # Check if the mapped source coordinates are within the valid range of the source image
#             if 0 <= x_src < src_img.shape[1] and 0 <= y_src < src_img.shape[0]:
#                 # Perform bilinear interpolation to compute the pixel value
#                 x1, y1 = int(x_src), int(y_src)
#                 x2, y2 = min(x1 + 1, src_img.shape[1] - 1), min(y1 + 1, src_img.shape[0] - 1)

#                 a = x_src - x1
#                 b = y_src - y1

#                 # Calculate the interpolated pixel value
#                 interpolated_value = (
#                     (1 - a) * (1 - b) * src_img[y1, x1] +
#                     a * (1 - b) * src_img[y1, x2] +
#                     (1 - a) * b * src_img[y2, x1] +
#                     a * b * src_img[y2, x2]
#                 )

#                 # Assign the computed pixel value to the output image
#                 warped_img[y, x] = interpolated_value

#     return warped_img


# In[ ]:


def draw_lines(points, img, title):
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
    cv2.imwrite(f'{title}_lines.jpg', img_copy)


# In[ ]:


def draw_points(points, img, title):
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
    cv2.imwrite(f'{title}_points.jpg', annotated_img)


# In[ ]:


def get_lines(point1, point2):
    point1= np.array([point1[0], point1[1], 1])
    point2= np.array([point2[0], point2[1], 1])
    line= np.cross(point1, point2)
    return line


# In[ ]:


def get_intersection(line1, line2):
    intersection= np.cross(line1, line2)
    return intersection


# In[ ]:


def correct_proj_distortion(image, point1, point2):
    vanishing_line= get_lines(point1, point2)
    vanishing_line /= np.linalg.norm(vanishing_line)
    # Create the homography matrix to align the vanishing line with the line at infinity
    H= np.identity(3, dtype=float)
    H[2]= vanishing_line # Set the third row to the vanishing line
    H= H/H[2,2]
    corrected_img= apply_homography(image, H)
    return corrected_img, H


# In[ ]:


def transform_lines(lines, H):
    lines_transformed= []
    for line in lines:
        line_trans= np.transpose(np.linalg.inv(H)) @ line
        line_trans /= np.linalg.norm(line_trans)
        lines_transformed.append(line_trans)
    return lines_transformed


# In[ ]:


def correct_affine_distortion(image, lines, H):
    # Transform the lines using the initial homography matrix
    lines_trans = transform_lines(lines, H)

    # solve for S
    X= np.zeros((2, 2), dtype= float)
    y= np.zeros((2, ), dtype=float)

    # Form the first linear equation from the transformed lines
    # first equation
    X[0, 0]= lines_trans[0][0]* lines_trans[1][0]
    X[0, 1]= lines_trans[0][0]* lines_trans[1][1] + lines_trans[0][1]* lines_trans[1][0]
    y[0]= lines_trans[0][1] * lines_trans[1][1]

    # second equation
    X[1, 0]= lines_trans[2][0]* lines_trans[3][0]
    X[1, 1]= lines_trans[2][0]* lines_trans[3][1] + lines_trans[2][1]* lines_trans[3][0]
    y[1]= lines_trans[2][1] * lines_trans[3][1]

    # Solve for the symmetric matrix S that represents the affine transformation
    s= np.linalg.inv(X) @ y
    S= np.ones((2, 2), dtype=float)
    S[0, 0]= s[0]
    S[0, 1]= s[1]
    S[1, 0]= S[0, 1]

    _, s, v= np.linalg.svd(S)
    eigenvalues= np.sqrt(np.diag(s))
    A= v @ eigenvalues @ np.transpose(v)
    H_from_s= np.zeros((3,3), dtype=float)
    H_from_s[0:2, 0:2]= A
    H_from_s[2, 2] = 1
    

    H_final= np.linalg.inv(H_from_s) @ H
    corrected_img= apply_homography(image, H_final)
    return corrected_img, H_final
    


# In[ ]:


def correct_distortion_one_step(image, lines):

    # Initialize the matrix to solve for distortion coefficients
    A = np.zeros((5, 5), dtype=float)

    # Separate line pairs into two sets (l_lines and m_lines)
    l = np.array([lines[0], lines[2], lines[4], lines[6], lines[8]])
    m = np.array([lines[1], lines[3], lines[5], lines[7], lines[9]])

    # Normalize line equations by their last coordinate
    l = (l.T / l[:, 2]).T
    m = (m.T / m[:, 2]).T

    # Fill the coefficients matrix with terms derived from line pairs
    A[:, 0] = l[:, 0] * m[:, 0]
    A[:, 1] = l[:, 0] * m[:, 1] + l[:, 1] * m[:, 0]
    A[:, 2] = l[:, 1] * m[:, 1]
    A[:, 3] = l[:, 0] + m[:, 0]
    A[:, 4] = l[:, 1] + m[:, 1]

    # Solve for the distortion coefficients vector
    d = np.dot(np.linalg.pinv(A), -np.ones((5, 1))).flatten()
    d /= np.max(d)

    # Construct the distortion correction matrix
    C = np.ones((3, 3))
    C[0, 0:2] = d[0:2]
    C[0, 2] = d[3]
    C[1, 0:2] = d[1:3]
    C[1, 2] = d[4]
    C[2, 0:2] = d[3:5]

    # Initialize the homography matrix
    H = np.zeros((3, 3))

    # Perform Singular Value Decomposition (SVD) on the 2x2 submatrix
    U, s, V_T = np.linalg.svd(C[0:2, 0:2])
    S = np.sqrt(np.diag(s))
    A_matrix = np.dot(np.dot(U, S), V_T)

    # Fill in the affine transformation part of the homography matrix
    H[0:2, 0:2] = A_matrix
    H[2, 0:2] = np.dot(np.linalg.pinv(A_matrix), C[0:2, 2]).T
    H[2, 2] = 1

    # Compute the inverse of the homography matrix for final correction
    H_final = np.linalg.inv(H)

    # Apply the computed homography to correct the image
    corrected_img = apply_homography(image, H_final)

    return corrected_img, H_final


# In[ ]:


board= cv2.imread('hw3/board_1.jpeg')
corridor= cv2.imread('hw3/corridor.jpeg')
tv= cv2.imread('hw3/tv.jpg')
wall= cv2.imread('hw3/wall.jpg')


# ### Select Image

# In[ ]:


selected= 'corridor'


# ### Point to Point Correspondence

# In[ ]:


#select points
if selected == 'board':
    point_image= np.array([[70, 421], [1222, 138], [1354, 1954], [420, 1791]])
    point_real= np.array([[300, 300], [800+300, 300], [800+300, 1200+300], [0+300, 1200+300]])
    image= board
if selected == 'corridor':
    point_image= np.array([[1084, 529], [1305, 487], [1296, 1340], [1078, 1216]])
    point_real= np.array([[600, 600], [300+600, 600], [300+600, 600+600], [0+600, 600+600]])
    image= corridor
if selected == 'tv':
    point_image= np.array([[688, 1521], [2645,1122], [2785,3060], [748, 2824]])
    point_real= np.array([[600, 600], [730+600, 600], [730+600, 430+600], [0+600, 430+600]])
    image= tv
if selected == 'wall':
    point_image= np.array([[2155, 2735], [2619,2750], [2733,3912], [2221, 3926]])
    point_real= np.array([[600, 600], [710+600, 600], [710+600, 330+600], [0+600, 330+600]])
    image= wall


# In[ ]:


H= compute_homography(point_image, point_real)


# In[ ]:


corrected_img= apply_homography(image, H)


# In[ ]:


plt.imshow(corrected_img)


# In[ ]:


cv2.imwrite(f'p2pcorrected{selected}.jpeg', corrected_img)


# ### Two Step Method

# In[ ]:


if selected == 'board':
    line1_1_points= np.array([[421, 1790], [70, 422]]) # vertical left
    line1_2_points= np.array([[1354, 1955], [1222, 139]]) # vertical right
    line2_1_points= np.array([[70, 422], [1222, 138]]) # horizontal up
    line2_2_points= np.array([[421, 1790], [1354, 1955]]) # horizontal down

    line1_1= get_lines(line1_1_points[0], line1_1_points[1])
    line1_2= get_lines(line1_2_points[0], line1_2_points[1])

    line2_1= get_lines(line2_1_points[0], line2_1_points[1])
    line2_2= get_lines(line2_2_points[0], line2_2_points[1])

    intersection_1= get_intersection(line1_1, line1_2)
    intersection_2= get_intersection(line2_1, line2_2)

    intersection_1= intersection_1/intersection_1[2]
    intersection_2= intersection_2/intersection_2[2]


# In[ ]:


if selected == 'corridor':
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


# In[ ]:


if selected == 'tv':
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


# In[ ]:


if selected == 'wall':
    point_image= np.array([[2155, 2735], [2619,2750], [2733,3912], [2221, 3926]])
    point_real= np.array([[600, 600], [710+600, 600], [710+600, 330+600], [0+600, 330+600]])
    image= wall

if selected == 'wall':
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


# In[ ]:


corrected_img, H= correct_proj_distortion(image, intersection_1, intersection_2)


# In[ ]:


plt.imshow(corrected_img)


# In[ ]:


cv2.imwrite(f'first_step_{selected}.jpeg', corrected_img)


# In[ ]:


lines = [line1_1, line2_1, line1_2, line2_2, line1_1, line2_2, line1_2, line2_1] # to be arranged such that every pair of lines is orthogonal in world 2d


# In[ ]:


corrected_img, H_final= correct_affine_distortion(image, lines, H)


# In[ ]:


cv2.imwrite(f'second_step_{selected}.jpeg', corrected_img)


# ### One step method

# In[ ]:


if selected== 'board':
# Additional lines
    line3_1_points= np.array([[1439, 2013], [1427, 564]]) # vertical rack
    line3_2_points= np.array([[1427, 564], [1510, 553]]) # horizontal rack

    line3_1= get_lines(line3_1_points[0], line3_1_points[1])
    line3_2= get_lines(line3_2_points[0], line3_2_points[1])

    


# In[ ]:


if selected=='corridor':
    line3_1_points= np.array([[815, 576], [920, 558]]) # vertical rack
    line3_2_points= np.array([[815, 576], [811, 1061]]) # horizontal rack

    line3_1= get_lines(line3_1_points[0], line3_1_points[1])
    line3_2= get_lines(line3_2_points[0], line3_2_points[1])


# In[ ]:


if selected=='tv':
    line3_1_points= np.array([[429, 3259], [1805, 3582]]) # vertical rack
    line3_2_points= np.array([[429, 3259], [479, 3925]]) # horizontal rack

    line3_1= get_lines(line3_1_points[0], line3_1_points[1])
    line3_2= get_lines(line3_2_points[0], line3_2_points[1])


# In[ ]:


if selected=='wall':
    line3_1_points= np.array([[2033, 2169], [2964, 2243]]) # vertical wall edge
    line3_2_points= np.array([[2033, 2169], [2055, 2633]]) # horizontal wall edge

    line3_1= get_lines(line3_1_points[0], line3_1_points[1])
    line3_2= get_lines(line3_2_points[0], line3_2_points[1])


# In[ ]:


line_points= [point for point in [line for line in [line1_1_points, line1_2_points, line2_1_points, line2_2_points, line3_1_points, line3_2_points]]]


# In[ ]:


lines.extend([line3_1, line3_2])


# In[ ]:


corrected_img, H= correct_distortion_one_step(image, lines)


# In[ ]:


cv2.imwrite(f'one_step_{selected}.jpeg', corrected_img)


# In[ ]:


# to plot adopted lines
draw_lines(line_points, image, selected)


# In[ ]:


# to plot adopted points
draw_points(point_image, image, selected)

