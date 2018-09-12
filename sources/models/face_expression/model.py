import cv2
import numpy as np

from keras.layers import Convolution2D, MaxPooling2D, Dropout, Conv2D
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

from sources.models.model import Models

class FaceExpressionModel(Models):

    def __init__(self):
        Models.__init__(self, 'FaceExpression')

    def build(self):
        print('build model', self.name)

        self.model.add(Conv2D(32, (2, 2), input_shape=(
            48, 48, 1), activation='relu', padding='same'))
        self.model.add(Conv2D(32,  (2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(.15))

        self.model.add(Conv2D(32,  (2, 2), activation='relu', padding='same'))
        self.model.add(Conv2D(32, (2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(.15))

        self.model.add(Conv2D(32,  (2, 2), activation='relu', padding='same'))
        self.model.add(Conv2D(32,  (2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(.15))

        self.model.add(Conv2D(32,  (2, 2), activation='relu', padding='same'))
        self.model.add(Conv2D(32,  (2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(.15))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(.15))
        self.model.add(Dense(7, activation='softmax'))
        return

   
    def predict(self, face):
        face_normalised = face / 255
        face_resized = cv2.resize(face_normalised, (48, 48), interpolation=cv2.INTER_AREA)
        face_resized = face_resized.reshape(48, 48, 1)
        face_resized = np.expand_dims(face_resized, axis=0)
        result = self.model.predict(face_resized)
        score = list(zip(['angry', 'disgust', 'fear', 'happy',
                          'sad', 'surprise', 'neutral'], result[0]))
        score.sort(key=lambda x: x[1], reverse=True)
        return score
    