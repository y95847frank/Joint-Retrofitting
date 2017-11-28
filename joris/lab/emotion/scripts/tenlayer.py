import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras import objectives
from keras import initializers

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

gamma = 1
batch_size = 16
latent_dim = 2048*4
epsilon_std = 0.0
epochs = 150
dropout_rate = 0.0
original_img_size= (192, 1024, 1)

inputs = Input(shape=original_img_size)
x = Conv2D(128,
           kernel_size=(5, 5),
           strides=(2, 2),
           padding='same')(inputs)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(128,
           kernel_size=(4, 4),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(128,
           kernel_size=(4, 4),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(256,
           kernel_size=(4, 4),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(256,
           kernel_size=(4, 4),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
'''
x = Conv2D(256,
           kernel_size=(4, 4),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(512,
           kernel_size=(4, 4),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(512,
           kernel_size=(4, 4),
           strides=(1, 2),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(512,
           kernel_size=(4, 4),
           strides=(1, 1),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(1024,
           kernel_size=(1, 1),
           strides=(1, 1),
           padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
'''

#sampling
z = Flatten()(x)
encoder = Model(inputs, z)

#decoder
x = Reshape((6, 32, 256))(z)
'''
x = Conv2DTranspose(1024,
                    kernel_size=(1, 1),
                    strides=(1, 1), 
                    padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(512,
                    kernel_size=(4, 4),
                    strides=(1, 1), 
                    padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(512,
                    kernel_size=(4, 4),
                    strides=(1, 1), 
                    padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(256,
                   kernel_size=(4, 4),
                   strides=(1, 1), 
                   padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(256,
                   kernel_size=(4, 4),
                   strides=(1, 2), 
                   padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(256,
                    kernel_size=(4, 4),
                    strides=(2, 2), 
                    padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
'''
x = Conv2DTranspose(256,
                    kernel_size=(4, 4),
                    strides=(2, 2), 
                    padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(256,
                    kernel_size=(4, 4),
                    strides=(2, 2), 
                    padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(128,
                    kernel_size=(5, 5),
                    strides=(2, 2), 
                    padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(128,
                    kernel_size=(5, 5),
                    strides=(2, 2), 
                    padding='same')(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
x = Conv2DTranspose(1,
                    kernel_size=(2, 2),
                    strides=(2, 2), 
                    padding='same',
                    activation='sigmoid')(x)
x_decoded_mean_squash = Reshape(original_img_size)(x)

def vae_loss(x, x_decoded_mean_squash):
    x = K.flatten(x)
    x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
    xent_loss = original_img_size[0] * original_img_size[1] * metrics.binary_crossentropy(x, x_decoded_mean_squash)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + gamma * kl_loss)

vae = Model(inputs, x_decoded_mean_squash)
vae.compile(optimizer='adam', loss='mean_squared_error')
model_json = vae.to_json()
with open('checkpoints/bigvec_model.json', 'w') as json_file:
    json_file.write(model_json)

# Training
x_train = np.load('/tmp/data/train_log.npy')
x_train = x_train[:,:,:1024]
x_train = x_train.reshape((x_train.shape[0], ) + original_img_size)

import h5py
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='checkpoints/bigvec_model.h5',
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min')
vae.fit(x_train, 
        x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint, TensorBoard(log_dir='/tmp/data')])
