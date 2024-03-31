import tensorflow as tf
from typing import Tuple


class ConvContinuousDecoder(tf.keras.Model):
    def __init__(self,
                 output_shape: tf.Tensor,
                 depth: int = 3
                 ):
        super().__init__()
        self._output_shape = output_shape

        self._input_layer = tf.keras.layers.Dense(32 * 32 * 16
                                                  , activation='relu')

        self._conv_layer1 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=4, strides=4, padding='same',
            activation='relu')

        self._conv_layer2 = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=4, strides=2, padding='same',
            activation='relu')

        self._mu_output = tf.keras.layers.Conv2DTranspose(
            filters=self._output_shape[-1], kernel_size=4, strides=2, padding='same')
        self._logvar_output = tf.keras.layers.Conv2DTranspose(
            filters=self._output_shape[-1], kernel_size=4, strides=2, padding='same')

    def call(self,
             z: tf.Tensor
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        z = self._input_layer(z)

        z = tf.reshape(z, (z.shape[0], 32, 32, 16))

        z = self._conv_layer1(z)
        z = self._conv_layer2(z)

        mu = self._mu_output(z)
        mu = tf.reshape(mu, (mu.shape[0]
                             , self._output_shape[0], self._output_shape[1], self._output_shape[2]))

        logvar = self._logvar_output(z)
        logvar = tf.reshape(logvar, (logvar.shape[0]
                                     , self._output_shape[0], self._output_shape[1], self._output_shape[2]))

        return mu, logvar

