'''
Created on 05 feb 2019

@author: Omar Cotugno
'''

from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from keras.models import Sequential


class AutoEncoder(object):

    def __init__(self, model_name, input_size, layers_size, kernel_size=(3, 3), is_rgba=False):

        if type(input_size) in [tuple, int]:
            self.model_name = model_name
            self.num_channels = 4 if is_rgba else 1
            self.model = self.load_model(model_name, input_size, layers_size, kernel_size)
        else:
            raise ValueError('input_size should be either tuple or int')

    def __autoencoder(self, input_size, layers_size):

        if type(input_size) == tuple:
            w = input_size[0]
            h = input_size[1]
            c = self.num_channels
            input_shape = (w * h * c,)
        elif type(input_size) == int:
            input_shape = input_size

        model = Sequential()

        for i, l in enumerate(layers_size):
            if i == 0:
                model.add(Dense(l, activation='relu', input_shape=input_shape))
            else:
                model.add(Dense(l, activation='relu'))

        model.add(Dense(input_shape[0], activation='sigmoid'))

        return model

    def __deep_autoencoder(self, input_size, layers_size):

        if type(input_size) == tuple:
            w = input_size[0]
            h = input_size[1]
            c = self.num_channels
            input_shape = (w * h * c,)
        elif type(input_size) == int:
            input_shape = input_size

        model = Sequential()

        for i, l in enumerate(layers_size):
            if i == 0:
                model.add(Dense(l, activation='relu', input_shape=input_shape))
            else:
                model.add(Dense(l, activation='relu'))

        model.add(Dense(input_shape[0], activation='sigmoid'))

        return model

    def __convolutional_autoencoder(self, input_size, kernel_size):

        input_shape = input_size + (self.num_channels,)

        model = Sequential()
        model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPool2D(padding='same'))
        model.add(Conv2D(16, kernel_size, activation='relu', padding='same'))
        model.add(MaxPool2D(padding='same'))
        model.add(Conv2D(8, kernel_size, activation='relu', padding='same'))
        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size, activation='relu', padding='same'))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
        model.add(Conv2D(self.num_channels, kernel_size, activation='sigmoid', padding='same'))

        return model

    def __convolutional_autoencoder_raw(self, input_size, kernel_size):

        input_shape = input_size + (self.num_channels,)

        model = Sequential()
        model.add(Conv2D(16, kernel_size, activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPool2D(padding='same'))
        model.add(Conv2D(8, kernel_size, activation='relu', padding='same'))
        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size, activation='relu', padding='same'))
        model.add(Conv2D(self.num_channels, kernel_size, activation='sigmoid', padding='same'))

        return model

    def load_model(self, name, input_size, layers_size, kernel_size):

        if name == 'autoencoder':
            return self.__autoencoder(input_size, layers_size)
        elif name == 'deep_autoencoder':
            return self.__deep_autoencoder(input_size, layers_size)
        elif name == 'convolutional_autoencoder':
            return self.__convolutional_autoencoder(input_size, kernel_size)
        elif name == 'convolutional_autoencoder_raw':
            return self.__convolutional_autoencoder_raw(input_size, kernel_size)
        else:
            raise ValueError('Unknown model name %s was given' % name)
