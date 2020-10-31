#!/usr/bin/env python

import cv2
import numpy as np
# Read Image

def estimate_head_position(image, image_points):
    size = image.shape
    # 3D model points, these are hyperparameters
    # image_points = np.array([
    #                         (359, 391),     # Nose tip
    #                         (399, 561),     # Chin
    #                         (337, 297),     # Left eye left corner
    #                         (513, 301),     # Right eye right corne
    #                         (345, 465),     # Left Mouth corner
    #                         (453, 469)      # Right mouth corner
    #                     ], dtype="double")
    image_points = image_points.astype(np.float)
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])
    
    # Estimated camera parameters, ToDo: Calibrate my camera and change them.
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
                            [[focal_length, 0           , center[0]],
                            [0            , focal_length, center[1]],
                            [0            , 0           , 1]], dtype = "double")

    # ToDo : change this later via calibration 
    dist_coeffs = np.zeros((4, 1))
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    for p in image_points:
        cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(image, p1, p2, (255,0,0), 2)
    
    # Display image
    return image