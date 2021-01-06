import numpy as np
import cv2
import imutils
from imutils import face_utils
from skimage import transform as tf

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def compute_eye_boxes(eye_feats, bias = np.int(0)):
    xmin, ymin = eye_feats.min(axis = 0).astype(int)
    xmax, ymax = eye_feats.max(axis = 0).astype(int)

    xmin -= bias
    xmax += bias

    ymin -= bias
    ymax += bias
    return (xmin, xmax, ymin, ymax)

def plot_eye_boxes(image, eye_feats, bias=0, verbose=False):
    xmin, xmax, ymin, ymax = compute_eye_boxes(eye_feats, bias)
    if verbose:
        #print("Eye shape : {}x{}".format(ymax - ymin, xmax - xmin))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), thickness=3)
    return image[ymin:ymax, xmin:xmax, :]

def rotate_eye(eye, angle):
    n, m, c = eye.shape
    # Assume that center of the eye (in general case it's not a pupil)
    # is in the center of detected bounding box
    rot_mat = cv2.getRotationMatrix2D((m // 2, n // 2), angle, 1)
    eye = cv2.warpAffine(eye, rot_mat, (m, n))
    return eye

def show_corners(image, shape, verbose=False):
    nose_tip = shape[30]
    chin = shape[8]
    left_eye_left_corner = shape[36]
    right_eye_right_corner = shape[45]
    left_mouth = shape[48]
    right_mouth = shape[54]

    index_array = [30, 8, 36, 45, 48, 54]
    if verbose:
        for index in index_array:
            cv2.circle(image, (shape[index][0], shape[index][1]), 1, (0, 255, 0), thickness=2)
    return shape[index_array]


class FaceAligner:
    def __init__(self, predictor, desiredLeftEye = (0.3, 0.3),
        desiredFaceWidth = 224, desiredFaceHeight = 224):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
            
    def align(self, image, gray, rect, bias=20,verbose=False):
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEyePart = shape[36:42]
        rightEyePart = shape[42:48]
        
        corners = show_corners(image, shape, verbose=verbose)
        if verbose:
            for (x, y) in np.vstack((leftEyePart, rightEyePart)):
                cv2.circle(image, (x, y), 1, (255, 255, 0), thickness=1)
                
        leftEyeCenter = leftEyePart.mean(axis = 0).astype(int)
        rightEyeCenter = rightEyePart.mean(axis = 0).astype(int)
        
        if verbose:
            cv2.line(image, tuple(leftEyeCenter), tuple(rightEyeCenter), (0, 255, 255), thickness=1)
        # cv2.imshow("EyesLandmarks", image)
        
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        
        angle = np.degrees(np.arctan2(dY, dX))
        
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        # eyesCenter = (shape[[37, 40, 32]].mean(), shape[[36, 43, 46]].mean())
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
                      
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        if verbose:
            cv2.circle(image, eyesCenter, 3, (255, 255, 0), thickness=1)
        
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # left_eye = plot_eye_boxes(image, leftEyePart, bias=bias, verbose=verbose)
        # right_eye = plot_eye_boxes(image, rightEyePart, bias=bias, verbose=verbose)
        return output, eyesCenter, M
        
