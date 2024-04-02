import tensorflow as tf
from typing import Tuple
from ResidualBlock import ResidualBlock

class ConvEncoder(tf.keras.Model):
    def __init__(self,
                 latent_size: int,
                 ):
        super().__init__()
        self._latent_size = latent_size

        self._conv_layer1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=2, strides=2,
            activation='relu', kernel_initializer='glorot_normal')

        self._conv_layer2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=2, strides=2,
            activation='relu', kernel_initializer='glorot_normal')

        self._res_block1 = ResidualBlock(64)

        self._conv_layer3 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=2, strides=2,
            activation='relu', kernel_initializer='glorot_normal')

        self._conv_layer4 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=2, strides=2,
            activation='relu', kernel_initializer='glorot_normal')

        self._res_block2 = ResidualBlock(256)

        self._flatten_layer = tf.keras.layers.Flatten()

        self._mean_output = tf.keras.layers.Dense(self._latent_size)
        self._std_output = tf.keras.layers.Dense(self._latent_size)

    def call(self,
             x: tf.Tensor
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self._conv_layer1(x)
        x = self._conv_layer2(x)
        x = self._res_block1(x)

        x = self._conv_layer3(x)
        x = self._conv_layer4(x)
        x = self._res_block2(x)

        x = self._flatten_layer(x)

        mean = self._mean_output(x)
        std = self._std_output(x)

        return mean, std