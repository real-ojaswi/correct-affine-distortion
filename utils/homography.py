import numpy as np
import cv2
import matplotlib.pyplot as plt

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