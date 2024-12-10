import numpy as np
from typing import List

from utils.utils import get_lines, get_intersection, transform_lines
from utils.homography import compute_homography, apply_homography

class CorrectDistortion:
    def __init__(self):
        pass

    def correct_affine_distortion(self, image, lines, H):
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
    

    def correct_proj_distortion(self, image, point1, point2):
        """
        Input:
        image: Image to be corrected
        point1: First vanishing point in the image
        point2: Second vanishing point in the image

        Output:
        corrected_img: Image after correction of projective distortion
        H: Homography needed to correct projective distortion
        """
        vanishing_line= get_lines(point1, point2)
        vanishing_line /= np.linalg.norm(vanishing_line)
        # Create the homography matrix to align the vanishing line with the line at infinity
        H= np.identity(3, dtype=float)
        H[2]= vanishing_line # Set the third row to the vanishing line
        H= H/H[2,2]
        corrected_img= apply_homography(image, H)
        return corrected_img, H
    

    def correct_distortion_two_step(self, image, point1, point2, lines: List):
        """
        Input:
        image: Image to be corrected
        point1: First vanishing point in the image
        point2: Second vanishing point in the image
        lines: List of homogenous coordinate representation of lines such that every pair in the list represents orthogonal lines
        (Atleast 4 pairs are needed)

        Output:
        corrected_img_first_step: Image after correction of projective distortion
        H: Homography needed to correct projective distortion
        corrected_img_second_step: Image after correction of both projective and affine distortion
        H: Homography needed to correct both projective and affine distortion
        """
        corrected_img_first_step, H= self.correct_proj_distortion(image, point1, point2)
        corrected_img_second_step, H_final= self.correct_affine_distortion(image, lines, H)

        return corrected_img_first_step, H, corrected_img_second_step, H_final


    def correct_distortion_one_step(self, image, lines):
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
    
    def correct_distortion_p2p(self, image, points1, points2):
        """
        Input:
        image: Image to be corrected
        points1: List of points (coordinates) in image
        points2: List of corresponding points (coordinates) in real world

        Output:
        corrected_img: Image after correction of projective and affine distortion
        H: Homography needed to correct projective and affine distortion
        """

        H= compute_homography(points1, points2)
        corrected_img= apply_homography(image, H)

        return corrected_img, H




