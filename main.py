import cv2

from sources.models.face_expression.model import FaceExpressionModel
from sources.utils.camera import Camera
from sources.utils.face import FaceDetection
from sources.utils.draw import face_square, draw_text


model = FaceExpressionModel()
model.load()
camera = Camera()

while True:
    frame = camera.get_frame()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FaceDetection().getInstance().detect(frame)
    for face in faces:
        x, y, w, h = face
        face_square(face, rgb_frame, (0, 0, 255))
        gray_face = gray_frame[y: y + h, x: x + w]
        score = model.predict(gray_face)
        draw_text([100, 100], rgb_frame, score[0][0], (0, 0, 255), font=0.5)
    
    cv2.imshow("Facial expression", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.stop()
cv2.destroyAllWindows()


