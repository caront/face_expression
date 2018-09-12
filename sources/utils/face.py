import cv2
import os

class FaceDetection():
    __instance = None

    @staticmethod
    def getInstance():
        if FaceDetection.__instance == None:
            FaceDetection()
        return FaceDetection.__instance

    def __init__(self):
        # if FaceDetection.__instance != None:
        #     raise Exception('use FaceDetection.getInstance()')
        # else:
        self.face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'classifier', 'haarcascade_frontalface_default.xml'))
        FaceDetection.__instance=self
    
    def detect(self, frame):
        return self.face_cascade.detectMultiScale(frame, 1.25, 6)


   