from keras.models import Model
from keras.layers import Input, merge, AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

# Modular function for Fire Node

def fire_module(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, 1, 1, border_mode='valid')(x)
    x = Activation('relu')(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid')(x)
    left = Activation('relu')(left)

    right= ZeroPadding2D(padding=(1, 1))(x)
    right = Convolution2D(expand, 3, 3, border_mode='valid')(right)
    right = Activation('relu')(right)

    x = merge([left, right], mode='concat', concat_axis=1)
    return x


# Original SqueezeNet from paper. Global Average Pool implemented manually

def get_squeezenet(nb_classes):

    input_img = Input(shape=(3, 227, 227))
    x = Convolution2D(96, 7, 7, subsample=(2, 2), border_mode='valid')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 32, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 64, 256)
    x = Dropout(0.5)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(nb_classes, 1, 1, border_mode='valid')(x)

    # global pooling not available
    x = AveragePooling2D(pool_size=(15, 15))(x)
    x = Flatten()(x)
    out = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=input_img, output=[out])
    return model


# Experimental network design for small images !

def get_small_squeezenet(nb_classes):

    input_img = Input(shape=(3, 32, 32))
    x = Convolution2D(16, 3, 3, border_mode='same')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = fire_module(x, 64, 256)
    x = Dropout(0.5)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(nb_classes, 1, 1, border_mode='valid')(x)

    # global pooling not available
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    out = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=input_img, output=[out])
    return model



if __name__ == '__main__':

    model1 = get_squeezenet(1000)
    model2 = get_small_squeezenet(10)
