import cv2


class Camera():

    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def get_frame(self):
        return self.video_capture.read()[1]

    def stop(self):
        self.video_capture.release()

    
