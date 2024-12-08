# **Correct Affine and Projective Distortions**

This repository contains Python implementations to correct affine and projective distortions in images using homography transformations and geometric principles from scratch without using opencv or other libraries (as much as possible). The project includes one-step and two-step method to align images by removing distortions caused by perspective and affine transformations.

---

## **Features**

1. **Homography Matrix Computation**:
   - Calculates a homography matrix using Direct Linear Transformation (DLT) for point-to-point correspondence.

2. **Two-Step Distortion Correction**:
   - **Step 1**: Corrects projective distortion by aligning vanishing points to infinity.
   - **Step 2**: Corrects affine distortion using orthogonal constraints on transformed lines.

3. **One-Step Distortion Correction**:
   - Combines projective and affine correction using line constraints in a single step.

4. **Visualization Utilities**:
   - Annotates key points and lines on images for debugging and visualization.

5. **Applications**:
   - Corrects geometric distortions in real-world images.

---


### **Dependencies**
Ensure the following Python libraries are installed:
- `numpy`
- `opencv-python`
- `matplotlib`


