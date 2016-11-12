from keras.models import Model, model_from_json, Sequential
from keras.layers import Input, merge, AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2, activity_l2, l1
import sklearn.metrics as metrics
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Embedding, Input, merge, ELU
from keras.layers import Embedding, Input, merge, ELU
import numpy as np

ndata = 0
# frame size
nrows = 64
ncols = 64
wr = 0.00000
dp = 0.


# Modular function for Fire Node

def fire_module(x, squeeze=16, expand=64, trainable = True):
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

def get_squeezenet_trans( img_size = (64,64)):
    input_img = Input(shape=(3, img_size[0], img_size[1]))
    x = Convolution2D(96, 7, 7, subsample=(2, 2), border_mode='valid', trainable=False)(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 32, 128)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 64, 256)
    x = Dropout(0.5)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)

    # Otto takes over
    # speed, accel, distance, angle
    real_in = Input(shape=(2,), name='real_input')

    flat = Flatten()(x)
    M = merge([flat,real_in], mode='concat', concat_axis=1)
    D1 = Dense(16,W_regularizer=l2(wr), init='lecun_uniform')(M)
    ED1 = ELU()(D1)
    DED1 = Dropout(dp)(ED1)

    S1 = Dense(64,W_regularizer=l2(wr), init='lecun_uniform')(DED1)
    ES1 = ELU()(S1)

    Steer_out = Dense(1, activation='linear', name='steer_out', init='lecun_uniform')(ES1)

    model = Model(input=[real_in, input_img], output=[Steer_out])
    return model

def get_squeezenet_mini_trans( img_size = (64,64)):
    input_img = Input(shape=(3, img_size[0], img_size[1]))
    x = Convolution2D(96, 7, 7, subsample=(2, 2), border_mode='valid')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Otto takes over
    # speed, accel, distance, angle
    real_in = Input(shape=(2,), name='real_input')

    flat = Flatten()(x)
    M = merge([flat,real_in], mode='concat', concat_axis=1)
    D1 = Dense(256,W_regularizer=l2(wr), init='lecun_uniform')(M)
    ED1 = ELU()(D1)
    DED1 = Dropout(dp)(ED1)

    S1 = Dense(64,W_regularizer=l2(wr), init='lecun_uniform')(DED1)
    ES1 = ELU()(S1)

    Steer_out = Dense(1, activation='linear', name='steer_out', init='lecun_uniform')(ES1)

    model = Model(input=[real_in, input_img], output=[Steer_out])
    return model


def get_pretrained_squeezenet(model_path, weights_path):

    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)
    return model
