import numpy as np
import tensorflow as tf

from scipy import interpolate
from models.model import Model, default_opt

from models.layers.subpixel import SubPixel1D, SubPixel1D_v2

from keras import backend as K
from keras.layers import Concatenate, add
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# ----------------------------------------------------------------------------

class AudioUNet(Model):
  """Generic tensorflow model training code"""

  def __init__(self, from_ckpt=False, n_dim=None, r=2,
               opt_params=default_opt, log_prefix='./run'):
    # perform the usual initialization
    self.r = r
    Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                   opt_params=opt_params, log_prefix=log_prefix)

  def create_model(self, n_dim, r):
    # load inputs
    X, _, _ = self.inputs
    tf.compat.v1.keras.backend.set_session(self.sess)

    with tf.name_scope('generator'):
      x = X
      L = self.layers
      # dim/layer: 4096, 2048, 1024, 512, 256, 128,  64,  32,
      # n_filters = [  64,  128,  256, 384, 384, 384, 384, 384]
      n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
      # n_filters = [  256,  512,  512, 512, 512, 1024, 1024, 1024]
      # n_filtersizes = [129, 65,   33,  17,  9,  9,  9, 9]
      # n_filtersizes = [31, 31,   31,  31,  31,  31,  31, 31]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
      downsampling_l = []

      print('building model...')

      # downsampling layers
      for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
          x = (Conv1D(filters=nf, kernel_size=fs,
                  activation='relu', padding='same', kernel_initializer="Orthogonal",
                  strides=2))(x)

          x = BatchNormalization()(x)
          x = LeakyReLU(0.2)(x)
          x = Dropout(rate=0.5)(x)

          print('D-Block: ', x.get_shape())
          downsampling_l.append(x)

      # bottleneck layer
      with tf.name_scope('bottleneck_conv'):
          x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], 
                  activation='relu', padding='same', kernel_initializer="Orthogonal",
                  strides=2))(x)

          x = BatchNormalization()(x)
          x = LeakyReLU(0.2)(x)
          x = Dropout(rate=0.5)(x)

      # upsampling layers
      for l, nf, fs, l_in in reversed(list(zip(range(L), n_filters, n_filtersizes, downsampling_l))):
        with tf.name_scope('upsc_conv%d' % l):
          # (-1, n/2, 2f)
          x = (Conv1D(filters=2*nf, kernel_size=fs, 
                  activation='relu', padding='same',kernel_initializer="Orthogonal"))(x)

          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          x = Dropout(rate=0.5)(x)

          # (-1, n, f)
          x = SubPixel1D(x, r=2)
          x = Concatenate(axis=-1)([x, l_in])
          # (-1, n, 2f)

          print ('U-Block: ', x.get_shape())

      # final conv layer
      with tf.name_scope('lastconv'):
        x = Conv1D(filters=2, kernel_size=9,
                activation=None, padding='same', kernel_initializer=normal_init)(x)
        x = SubPixel1D(x, r=2)
        print(x.get_shape())

      g = add([x, X])

    return g

  def predict(self, X):
    assert len(X) == 1
    x_sp = spline_up(X, self.r)
    # x_sp = X[0]
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
    X = x_sp.reshape((1,len(x_sp),1))
    feed_dict = self.load_batch((X,X), train=False)
    return self.sess.run(self.predictions, feed_dict=feed_dict)

# ----------------------------------------------------------------------------
# helpers

def normal_init(shape, dtype=None):
  return K.random_normal(shape, stddev=1e-3, mean=0)


def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp