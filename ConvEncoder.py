import tensorflow as tf
from typing import Tuple


class ConvEncoder(tf.keras.Model):
    def __init__(self,
                 latent_size: int,
                 ):
        super().__init__()
        self._latent_size = latent_size

        self._conv_layer1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=4, strides=4,
            activation='relu', kernel_initializer='glorot_normal')

        self._conv_layer2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=4, strides=4,
            activation='relu', kernel_initializer='glorot_normal')

        self._flatten_layer = tf.keras.layers.Flatten()

        self._mean_output = tf.keras.layers.Dense(self._latent_size)
        self._std_output = tf.keras.layers.Dense(self._latent_size)

    def call(self,
             x: tf.Tensor
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self._conv_layer1(x)
        x = self._conv_layer2(x)
        x = self._flatten_layer(x)

        mean = self._mean_output(x)
        std = self._std_output(x)

        return mean, std


