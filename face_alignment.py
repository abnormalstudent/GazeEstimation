from FaceAligner import FaceAligner
from FaceAligner import rect_to_bb
from head_pose_esimation import estimate_head_position

import argparse
import imutils
import dlib
import cv2
import time
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
help = "Path to facial landmark predictor")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

video = cv2.VideoCapture(0)

start_time = time.time()
frames = 0
while(True):
    ret, image = video.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        #print("Head size : {}x{}\n".format(w, h))
        faceOrig = image[y : y + h, x : x + w]
        faceAligned, left_eye, right_eye, eyesCenter, corners = fa.align(image, gray, rect, verbose=True)
        #print(faceAligned.shape)
        # These values (36x60) are actually hyperparameters
        WIDTH, HEIGHT = 60, 36
        left_eye = cv2.resize(left_eye, (WIDTH, HEIGHT))
        right_eye = cv2.resize(right_eye, (WIDTH, HEIGHT))
        
        estimate_head_position(image, corners)
        cv2.imshow("Input", image)
        cv2.imshow("Original", faceOrig)
        cv2.imshow("Aligned", faceAligned)
        
        # Since eyes are inversed
        cv2.imshow("RightEye", left_eye)
        cv2.imshow("LeftEye", right_eye)

    key = cv2.waitKey(1) & 0xFF
    frames += 1
    if key == ord('q'):
        break
print("FPS : {}".format(frames / (time.time() - start_time)))
    