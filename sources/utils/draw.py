import cv2

def draw_text(coor, image_array, text, color, x_off=0, y_off=0, font=2):
    x, y = coor[:2]
    cv2.putText(image_array, text, (x+x_off, y + y_off),
                cv2.FONT_HERSHEY_SIMPLEX, font, color, 2, cv2.LINE_AA)

def face_square(coor, image_array, color):
    x, y, w, h = coor
    cv2.rectangle(image_array, (x, y), (x + w, y+h), color, 2)

