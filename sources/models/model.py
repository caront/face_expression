import abc
import os
from keras.models import Sequential
from keras.models import load_model

model_path = 'models_save'

class Models():
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name
        self.path = os.path.join(os.getcwd(), model_path, name)
        self.model = Sequential()

    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractclassmethod
    def exe_frame(self)
        pass

    @abc.abstractclassmethod
    def exe_image(self, image_path):
        pass

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, train, test, epochs=1000, steps_per_epoch=100, validation_steps=2000):
        return self.model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=test, validation_steps=validation_steps, verbose=1)

    def save(self, filename='model'):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.model.save(self.path + '/' + filename + '.h5')
        print('[x] Model', self.name, 'saved')

    def load(self, filename='model'):
        self.model = load_model(self.path + '/' + filename + '.h5')
        print('[x] Model', self.name, 'load')

    def Model(self):
        return self.model
