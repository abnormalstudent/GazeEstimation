import cv2
import numpy as np
import h5py
import os

def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def apply_affine_transform(points, transform):
    transform = transform.transpose()
    points = np.hstack([points, 1])
    return np.dot(points, transform)

def draw_gaze(image_in, pitchyaw, thickness=2, prediction=False, offset=(0, 0), transform=None, image_shape=None, original=False):
    """Draw gaze angle on given image with a given eye positions."""
    if prediction:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    image_out = image_in
    if image_shape is None:
        (h, w) = image_in.shape[:2]
    else:
        (h, w, _) = image_shape
    length = w / 2.0

    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])

    if original:
        pos = (int(offset[0]), int(offset[1]))
    else:
        pos = (int(h / 2.0 + offset[0]), int(w / 2.0 + offset[1]))
    if transform is not None:
        # if transform.shape == (2, 3):
        #     transform = np.vstack((transform, [0, 0, 1]))
        # print(transform.shape)
        # pos = apply_affine_transform(pos, transform)
        (dx, dy) = apply_affine_transform(np.array([dx, dy]), transform)

    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out
