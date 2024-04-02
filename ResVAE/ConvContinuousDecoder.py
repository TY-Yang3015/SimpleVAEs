import tensorflow as tf
from typing import Tuple
from ResidualBlock import ResidualBlock


class ConvContinuousDecoder(tf.keras.Model):
    def __init__(self,
                 output_shape: tf.Tensor,
                 depth: int = 3
                 ):
        super().__init__()
        self._output_shape = output_shape

        self._input_layer = tf.keras.layers.Dense(16 * 16 * 16
                                                  , activation='relu')

        self._conv_layer1 = tf.keras.layers.Conv2DTranspose(
            filters=128, kernel_size=2, strides=2, padding='same',
            activation='relu', kernel_initializer='glorot_normal')

        self._res_block1 = ResidualBlock(128)

        self._conv_layer2 = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=2, strides=2, padding='same',
            activation='relu', kernel_initializer='glorot_normal')

        self._reconstruct_output = tf.keras.layers.Conv2DTranspose(
            filters=self._output_shape[-1], kernel_size=2, strides=2, padding='same')


    def call(self,
             z: tf.Tensor
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        z = self._input_layer(z)

        z = tf.reshape(z, (z.shape[0], 16, 16, 16))

        z = self._conv_layer1(z)

        z = self._res_block1(z)

        z = self._conv_layer2(z)

        reconstruction = self._reconstruct_output(z)
        reconstruction = tf.reshape(reconstruction, (reconstruction.shape[0]
                             , self._output_shape[0], self._output_shape[1], self._output_shape[2]))


        min_val = tf.reduce_min(reconstruction)
        max_val = tf.reduce_max(reconstruction)
        reconstruction = (reconstruction - min_val) / (max_val - min_val + 1e-5)


        return reconstruction * 255.