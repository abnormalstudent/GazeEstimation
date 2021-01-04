# Local packages
from blazeface import BlazeFace

# PyTorch models handeling
import torch

# cv2.resize
from cv2 import resize

import numpy as np

def BlazeFace_loader(use_gpu=False, device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
    model = BlazeFace().to(device)
    model.load_weights("weights/blazeface.pth")
    model.load_anchors("weights/anchors.npy")
    return model

def round_keypoints(keypoints, x_scale, y_scale):
    ymin, xmin, ymax, xmax = keypoints
    return np.int(np.rint(ymin * y_scale * 128)), \
           np.int(np.rint(xmin * x_scale * 128)), \
           np.int(np.rint(ymax * y_scale * 128)), \
           np.int(np.rint(xmax * x_scale * 128))

def predict(model, image):
    height, width, _ = image.shape
    x_scale = width / 128
    y_scale = height / 128
    image = resize(image, (128, 128))

    predictions = model.predict_on_image(image)
    any_faces = predictions.shape[0] != 0
    if any_faces:
        predictions = predictions[0][:4]
    return None if not any_faces else round_keypoints(predictions, x_scale, y_scale)
