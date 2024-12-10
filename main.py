import cv2
import numpy as np
import sys
import os

from utils.correct_distortion import CorrectDistortion
from utils.utils import get_intersection, get_lines
from utils.plot_utils import draw_lines, draw_points
from utils.prepare_data import return_data

SELECTED= 'board'

def main(SELECTED, output_dir):
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    image, point_image, point_real, intersection_1, intersection_2, lines, extra_lines, line_points = return_data(SELECTED)

    correct = CorrectDistortion()

    corrected_image_p2p, H = correct.correct_distortion_p2p(image, point_image, point_real)
    output_path_p2p = os.path.join(output_dir, f"{SELECTED}_corrected_p2p.jpg")
    cv2.imwrite(output_path_p2p, corrected_image_p2p)


    corrected_image_first_step, H_first_step, corrected_image_second_step, H_second_step = correct.correct_distortion_two_step(image, intersection_1, intersection_2, lines)
    output_path_first_step = os.path.join(output_dir, f"{SELECTED}_corrected_first_step.jpg")
    cv2.imwrite(output_path_first_step, corrected_image_first_step)
    output_path_second_step = os.path.join(output_dir, f"{SELECTED}_corrected_second_step.jpg")
    cv2.imwrite(output_path_second_step, corrected_image_second_step)

    lines.extend(extra_lines)
    corrected_image_one_step, H = correct.correct_distortion_one_step(image, lines)
    output_path_one_step = os.path.join(output_dir, f"{SELECTED}_corrected_one_step.jpg")
    cv2.imwrite(output_path_one_step, corrected_image_one_step)

    # Optionally, draw and save the lines and points as well
    draw_lines(line_points, image, SELECTED, output_dir)
    draw_points(point_image, image, SELECTED, output_dir)

    print(f"Outputs saved at {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <selected_image_type> <output_directory>")
        sys.exit(1)
    
    SELECTED = sys.argv[1]
    output_dir = sys.argv[2]

    main(SELECTED, output_dir)
