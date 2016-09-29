from keras.models import Model, model_from_json, Sequential
from keras.layers import Input, merge, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
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

    y = merge([left, right], mode='concat', concat_axis=1)
    return y


# Original SqueezeNet from paper. Global Average Pool implemented manually with Average Pooling Layer

def get_squeezenet(nb_classes, img_size = (64,64)):

    input_img = Input(shape=(3, img_size[0], img_size[1]))
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
    x = GlobalAveragePooling2D()(x)
    out = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=input_img, output=[out])
    return model


def get_pretrained_squeezenet(model_path, weights_path):

    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)
    return model


