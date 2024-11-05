"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

import numpy as np

def getAffineM(src,dst):
    A = []
    B = []

    for (x, y), (x_prime, y_prime) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(x_prime)
        B.append(y_prime)

    A = np.array(A)
    B = np.array(B)
    affine_params = np.linalg.solve(A, B)
    M = affine_params.reshape(2, 3)
    return M

def applyAffine(src,M):
    dst = np.concatenate((src,np.ones((src.shape[0],1))),axis=1)
    dst = dst.astype(np.float32)
    dst = np.dot(dst,M.T)
    return dst

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
# Set Width and Height
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Frame rate
webcam.set(cv2.CAP_PROP_FPS, 30)

width  = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps    = webcam.get(cv2.CAP_PROP_FPS)
print(f'Width: {width}, Height: {height}, FPS: {fps}')

init_frame_count        = 30
calibration_frame_count = 90
calibrate_frame_width   = 1650
calibrate_frame_height  = 1000
calibrating_pins        = [[0.9,0.1],[0.1,0.1],[0.1,0.9]]
calibrating_points      = np.array(calibrating_pins) * np.array([calibrate_frame_height,calibrate_frame_width])
calibrating_points      = calibrating_points.astype(np.int32)
calibrate_point_num     = calibrating_points.shape[0]

print(calibrating_points)

pupil_points_left       = np.zeros_like(calibrating_points)
pupil_points_right      = np.zeros_like(calibrating_points)

# FSM states
frame_counter = 0
initialized   = 0
calibrating   = 0
calibrate_cnt = 0

pupil_data_points_left  = np.zeros((calibration_frame_count,2))
pupil_data_points_right = np.zeros((calibration_frame_count,2))

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)
    
    if not initialized:
        frame = gaze.annotated_frame()

        if (not initialized) and (not calibrating):
            text = "Initializing:" + str(frame_counter)
            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        cv2.imshow("Demo", frame)
    elif calibrating:
        ref_frame = np.ones((calibrate_frame_height, calibrate_frame_width), np.uint8) * 255

        color = (0,0,0)
        y = calibrating_points[calibrate_cnt,0]
        x = calibrating_points[calibrate_cnt,1]
        cv2.line(ref_frame, (x - 5, y), (x + 5, y), color)
        cv2.line(ref_frame, (x, y - 5), (x, y + 5), color)
        cv2.imshow("Reference", ref_frame)
        cv2.moveWindow("Reference", 0, 0)

        if gaze.pupils_located:
            pupil_data_points_left[frame_counter,0]  = gaze.eye_left.pupil.y
            pupil_data_points_left[frame_counter,1]  = gaze.eye_left.pupil.x
            pupil_data_points_right[frame_counter,0] = gaze.eye_right.pupil.y
            pupil_data_points_right[frame_counter,1] = gaze.eye_right.pupil.x
        else:
            pupil_data_points_left[frame_counter,0]  = -1
            pupil_data_points_left[frame_counter,1]  = -1
            pupil_data_points_right[frame_counter,0] = -1
            pupil_data_points_right[frame_counter,1] = -1
    else: # Done Calibrating
        ref_frame = np.ones((calibrate_frame_height, calibrate_frame_width), np.uint8) * 255

        # Left eye target
        color = (255,0,0)
        if gaze.pupils_located:
            dst = applyAffine(np.float32([[gaze.eye_left.pupil.y,gaze.eye_left.pupil.x]]),gaze.Affine_M_left)
        else:
            dst = np.int32([[calibrate_frame_height//2,calibrate_frame_width//2]])
        x, y  = int(dst[0,1]), int(dst[0,0])
        cv2.line(ref_frame, (x - 5, y), (x + 5, y), color)
        cv2.line(ref_frame, (x, y - 5), (x, y + 5), color)

        # Right eye target
        color = (0,0,255)
        if gaze.pupils_located:
            dst   = applyAffine(np.float32([[gaze.eye_right.pupil.y,gaze.eye_right.pupil.x]]),gaze.Affine_M_right)
        else:
            dst = np.int32([[calibrate_frame_height//2,calibrate_frame_width//2]])
        x, y  = int(dst[0,1]), int(dst[0,0])
        cv2.line(ref_frame, (x - 5, y), (x + 5, y), color)
        cv2.line(ref_frame, (x, y - 5), (x, y + 5), color)

        cv2.imshow("Reference", ref_frame)
        cv2.moveWindow("Reference", 0, 0)


    if cv2.waitKey(1) == 27:
        break

    # FSM Update Logic
    frame_counter += 1

    if (frame_counter == init_frame_count) and (not initialized) and (not calibrating):
        initialized   = 1
        frame_counter = 0
        calibrating   = 1

    if initialized and calibrating:
        if(frame_counter == calibration_frame_count):
            # Calculate the pupil_points
            masked_pupil_data_points_left  = np.ma.masked_array(pupil_data_points_left, mask=np.repeat(pupil_data_points_right[:,0] == -1,2))
            masked_pupil_data_points_right = np.ma.masked_array(pupil_data_points_right,mask=np.repeat(pupil_data_points_right[:,0] == -1,2))

            pupil_points_left[calibrate_cnt,:]  = np.mean(masked_pupil_data_points_left, axis=0)
            pupil_points_right[calibrate_cnt,:] = np.mean(masked_pupil_data_points_right,axis=0)
            pupil_points_left  = pupil_points_left.astype(np.float32)
            pupil_points_right = pupil_points_right.astype(np.float32)
            print("Left Pupil point: ",pupil_points_left[calibrate_cnt,:])
            print("Right Pupil point: ",pupil_points_right[calibrate_cnt,:])
            frame_counter  = 0
            calibrate_cnt += 1
            if(calibrate_cnt == calibrate_point_num):
                calibrating = 0
                # Calculate the affine transformation matrix
                Affine_M_left  = getAffineM(pupil_points_left,  calibrating_points.astype(np.float32))
                Affine_M_right = getAffineM(pupil_points_right, calibrating_points.astype(np.float32))
                gaze.set_Affine(Affine_M_left,Affine_M_right)
                print("Affine Left:",Affine_M_left)
                print("Affine Right:",Affine_M_right)
    

   
webcam.release()
cv2.destroyAllWindows()
