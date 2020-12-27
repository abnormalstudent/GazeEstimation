import torch

import cv2

from BlazeNet import BlazeFace_loader
from BlazeNet import predict

def main():
    blazeface = BlazeFace_loader(use_gpu=False)
    blazeface.eval()

    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        with torch.no_grad():
            any_faces = predict(blazeface, frame)
        if any_faces is not None:
            ymin, xmin, ymax, xmax = any_faces
        else:
            continue
        key = cv2.waitKey(1) & 0xFF
        frame = cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), (0, 0, 255), 3)

        # cv2.imshow('Detected face', frame[ymin : ymax, xmin : xmax])
        cv2.imshow('Detected face', frame)
        if key == ord('q'):
            break
if __name__ == '__main__':
    main()