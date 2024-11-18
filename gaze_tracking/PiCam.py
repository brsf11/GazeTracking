import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput

class OpenCVWebcam(object):
    def __init__(self,width=1280,height=960,fps=30):
        self.picam2 = Picamera2()
        self.video_config = self.picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"}) # Add fps setting?
        self.picam2.configure(self.video_config)
        self.encoder = H264Encoder(1000000, repeat=True)
        self.encoder.output = CircularOutput()
        self.picam2.start()
        self.picam2.start_encoder(self.encoder)

        # Need to get real settings. Opencv's webcam's actual setting can be different with the one we apply
        # For example when the resolution we try to set is larger than the supported one
        # Not sure if that's the case with PiCam

        # self.webcam_width  = self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.webcam_height = self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.webcam_fps    = self.webcam.get(cv2.CAP_PROP_FPS)

    def getSettings(self): # No need to change
        return self.webcam_width, self.webcam_height, self.webcam_fps
    
    def read(self): # Return one frame
        return True, self.picam2.capture_buffer("main").reshape((480,640,3)).astype(np.uint8)
    
    def __del__(self):
        #self.webcam.release()
        pass