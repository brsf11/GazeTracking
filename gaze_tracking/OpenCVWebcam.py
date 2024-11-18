import cv2

class OpenCVWebcam(object):
    def __init__(self,width=1280,height=960,fps=30):
        # Web cam
        self.webcam = cv2.VideoCapture(0)
        # Set Width and Height
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Frame rate
        self.webcam.set(cv2.CAP_PROP_FPS, fps)

        self.webcam_width  = self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.webcam_height = self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.webcam_fps    = self.webcam.get(cv2.CAP_PROP_FPS)

    def getSettings(self):
        return self.webcam_width, self.webcam_height, self.webcam_fps
    
    def read(self):
        return self.webcam.read()
    
    def __del__(self):
        self.webcam.release()