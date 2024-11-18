import cv2
from gaze_tracking.target_tracking import TgtTracking

target = TgtTracking()
target.show_webcam(5)

if not target.trans.init_calibration_done:
    target.initial_calibrate()

while True:
    target.refresh()

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('i'):
        print("Restart Calibration!")
        target.initial_calibrate()
