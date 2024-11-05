from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration
import numpy as np


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        pts = np.float32([[0,0],[0,1],[1,0]])
        self.Affine_M_left  = cv2.getAffineTransform(pts, pts)
        self.Affine_M_right = cv2.getAffineTransform(pts, pts)

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    def set_Affine(self,Affine_M_left,Affine_M_right):
        self.Affine_M_left  = Affine_M_left 
        self.Affine_M_right = Affine_M_right

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            # Pupils
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

            # Eye Region
            color = (255,0,0)
            for point in self.eye_left.landmark_points:
                x, y = point
                cv2.line(frame, (x - 5, y), (x + 5, y), color)
                cv2.line(frame, (x, y - 5), (x, y + 5), color)

            color = (0,0,255)
            for point in self.eye_right.landmark_points:
                x, y = point
                cv2.line(frame, (x - 5, y), (x + 5, y), color)
                cv2.line(frame, (x, y - 5), (x, y + 5), color)

            # Eye Direction
            color = (255, 255, 255)
            x_orig_left  = self.eye_left.origin[0]  + self.eye_left.center[0] - 5
            y_orig_left  = self.eye_left.origin[1]  + self.eye_left.center[1] - 5
            x_orig_right = self.eye_right.origin[0] + self.eye_right.center[0] - 5
            y_orig_right = self.eye_right.origin[1] + self.eye_right.center[1] - 5
            x_orig_left  = x_orig_left.astype(np.int32)
            y_orig_left  = y_orig_left.astype(np.int32)
            x_orig_right = x_orig_right.astype(np.int32)
            y_orig_right = y_orig_right.astype(np.int32)

            cv2.line(frame, (x_orig_left,  y_orig_left),  (x_orig_left +(10*(x_left- x_orig_left)),  y_orig_left +(10*(y_left -y_orig_left))),  color)
            cv2.line(frame, (x_orig_right, y_orig_right), (x_orig_right+(10*(x_right-x_orig_right)), y_orig_right+(10*(y_right-y_orig_right))), color)
            



        return frame
