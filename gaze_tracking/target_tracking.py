import cv2
# from gaze_tracking import GazeTracking
from .mediapipe_gaze import mediapipe_gaze
from .transformation import transformation_affine, transformation_perspective
from .OpenCVWebcam import OpenCVWebcam
import numpy as np
import os

def custom_clustering(data, threshold=0.01, max_iters=100):
    n, d = data.shape
    np.random.seed(42)
    
    labels = np.zeros(n, dtype=int)
    clean_data = data.copy()
    
    def gaussian(x, mean, cov):
        d = mean.shape[0]
        cov_inv = np.linalg.inv(cov)
        diff = x - mean
        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        return np.exp(exponent) / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    
    for _ in range(max_iters):
        mean = np.mean(clean_data, axis=0)
        cov = np.cov(clean_data, rowvar=False) + 1e-6 * np.eye(d)
        
        new_labels = np.zeros(n, dtype=int)
        for i in range(n):
            if gaussian(data[i], mean, cov) < threshold:
                new_labels[i] = -1
        
        if np.array_equal(new_labels, labels):
            break
        
        labels = new_labels
        clean_data = data[labels == 0] 
    
    return labels

def remove_outlier_3d(data,clustering=custom_clustering):
    P, n, d = data.shape
    label   = []
    for p in range(P):
        sub_data  = data[p,:,:]
        sub_label = clustering(sub_data)
        label.append(sub_label)

    label = np.array(label).reshape(P,n,1)
    label = np.repeat(label,d,axis=2)
    data = ((label == -1) * (np.ones_like(data) * -1)) + (((label == 0) * data))
    return data

def remove_outlier_2d(data,clustering=custom_clustering):
    n, d = data.shape
    label = clustering(data)
    label = np.array(label).reshape(n,1)
    label = np.repeat(label,d,axis=1)
    data = ((label == -1) * (np.ones_like(data) * -1)) + (((label == 0) * data))
    return data

class EMAFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.prev_target = np.array([None,None])

    def filter(self, target):
        if self.prev_target[0] is None or self.prev_target[1] is None:
            self.prev_target = target
        else:
            self.prev_target = (self.alpha * target) + ((1 - self.alpha) * self.prev_target)

        return self.prev_target

class TgtTracking(object):
    def __init__(self):
        self.gaze = mediapipe_gaze()

        # Initial calibration setting
        self.window_frame_width           = 1700
        self.window_frame_height          = 1000

        # Transformation
        self.trans = transformation_perspective()

        self.Camera = OpenCVWebcam(1280,960,30)
        self.webcam_width, self.webcam_height, self.webcam_fps = self.Camera.getSettings()

        # Filter
        self.filter_target = EMAFilter(alpha=0.1)

    def save_calib_pupil_points(self,pupil_data_points_left,pupil_data_points_right):
        np.savetxt("Pupil_debug_left.txt",  pupil_data_points_left.reshape(-1,  pupil_data_points_left.shape[2]))
        np.savetxt("Pupil_debug_right.txt", pupil_data_points_right.reshape(-1, pupil_data_points_right.shape[2]))
        pupil_data_points_left  = remove_outlier_3d(pupil_data_points_left)
        pupil_data_points_right = remove_outlier_3d(pupil_data_points_right)
        np.savetxt("Clean_Pupil_debug_left.txt",  pupil_data_points_left.reshape(-1,  pupil_data_points_left.shape[2]))
        np.savetxt("Clean_Pupil_debug_right.txt", pupil_data_points_right.reshape(-1, pupil_data_points_right.shape[2]))
    
    def show_text(self,frame,text):
        cv2.putText(frame, text, (self.window_frame_height//2, self.window_frame_width//2), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 2)
        return frame
    
    def show_webcam(self,duration=2):
        cnt = 0
        cv2.destroyAllWindows()
        while(cnt < (duration * self.webcam_fps)):
            _, frame = self.Camera.read()
            self.gaze.refresh(frame)
            webcam_frame = self.gaze.annotated_frame()
            cv2.imshow("webcam", webcam_frame)
            cv2.moveWindow("webcam", 0, 0)
            if cv2.waitKey(1) == ord('q'):
                break
            cnt += 1
        cv2.destroyAllWindows()

    def initial_calibrate(self):
        init_calibrating_points  = np.array(self.trans.init_calibrating_pins) * np.array([self.window_frame_height,self.window_frame_width])
        init_calibrating_points  = init_calibrating_points.astype(np.int32)
        init_calibrate_point_num = init_calibrating_points.shape[0]

        pupil_data_points_left  = np.zeros((init_calibrate_point_num,self.trans.init_calibration_frame_count,2))
        pupil_data_points_right = np.zeros((init_calibrate_point_num,self.trans.init_calibration_frame_count,2))
        calibrate_done = 0
        cv2.destroyAllWindows()
        while(not calibrate_done):
            for calibrate_cnt in range(init_calibrate_point_num):
                frame_cnt = 0
                while(frame_cnt < self.trans.init_calibration_frame_count):
                    _, frame = self.Camera.read()
                    self.gaze.refresh(frame)

                    # Render reference frame
                    ref_frame = np.ones((self.window_frame_height, self.window_frame_width), np.uint8) * 255

                    color = (0,0,0)
                    y = init_calibrating_points[calibrate_cnt,0]
                    x = init_calibrating_points[calibrate_cnt,1]
                    cv2.line(ref_frame, (x - 5, y), (x + 5, y), color)
                    cv2.line(ref_frame, (x, y - 5), (x, y + 5), color)
                    cv2.imshow("Reference", ref_frame)
                    cv2.moveWindow("Reference", 0, 0)

                    if self.gaze.pupils_located:
                        pupil_data_points_left[ calibrate_cnt,frame_cnt,0] = self.gaze.eye_left_pupil_y
                        pupil_data_points_left[ calibrate_cnt,frame_cnt,1] = self.gaze.eye_left_pupil_x
                        pupil_data_points_right[calibrate_cnt,frame_cnt,0] = self.gaze.eye_right_pupil_y
                        pupil_data_points_right[calibrate_cnt,frame_cnt,1] = self.gaze.eye_right_pupil_x
                        frame_cnt += 1
                    # else:
                    #     pupil_data_points_left[ calibrate_cnt,frame_cnt,0] = -1
                    #     pupil_data_points_left[ calibrate_cnt,frame_cnt,1] = -1
                    #     pupil_data_points_right[calibrate_cnt,frame_cnt,0] = -1
                    #     pupil_data_points_right[calibrate_cnt,frame_cnt,1] = -1


                    if cv2.waitKey(1) == ord('q'):
                        break
                    
            
            # After Data collection for all calibration pins, try to get the affine matrix
            # Calculate the pupil_points
            self.save_calib_pupil_points(pupil_data_points_left,pupil_data_points_right)
            # Remove outliers
            pupil_data_points_left  = remove_outlier_3d(pupil_data_points_left)
            pupil_data_points_right = remove_outlier_3d(pupil_data_points_right)

            masked_pupil_data_points_left  = np.ma.masked_array(pupil_data_points_left, mask=(pupil_data_points_left  == -1))
            masked_pupil_data_points_right = np.ma.masked_array(pupil_data_points_right,mask=(pupil_data_points_right == -1))

            pupil_points_left  = np.mean(masked_pupil_data_points_left, axis=1)
            pupil_points_right = np.mean(masked_pupil_data_points_right,axis=1)
            pupil_points_left  = pupil_points_left.astype(np.float32)
            pupil_points_right = pupil_points_right.astype(np.float32)
            print("Left Pupil point: ",pupil_points_left)
            print("Right Pupil point: ",pupil_points_right)

            # Calculate the affine transformation matrix
            calibrate_done = self.trans.init_calibrate(pupil_points_left,pupil_points_right,init_calibrating_points)
        cv2.destroyAllWindows()
    
    def refresh(self):
        _, frame = self.Camera.read()

        self.gaze.refresh(frame)
        ref_frame = np.ones((self.window_frame_height, self.window_frame_width), np.uint8) * 255

        # Left eye target
        color = (0,0,0)
        if self.gaze.pupils_located:
            dst = self.trans.transform(np.float32([self.gaze.eye_left_pupil_y, self.gaze.eye_left_pupil_x]),np.float32([self.gaze.eye_right_pupil_y,self.gaze.eye_right_pupil_x]))
        else:
            dst = np.int32([self.window_frame_height//2,self.window_frame_width//2])
        dst = self.filter_target.filter(dst)
        x, y  = int(dst[1]), int(dst[0])
        cv2.line(ref_frame, (x - 5, y), (x + 5, y), color)
        cv2.line(ref_frame, (x, y - 5), (x, y + 5), color)

        cv2.imshow("Reference", ref_frame)
        cv2.moveWindow("Reference", 0, 0)