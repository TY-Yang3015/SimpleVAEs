import tensorflow as tf
from typing import Tuple


class Encoder(tf.keras.Model):
    def __init__(self,
                 hidden_size: int,
                 latent_size: int,
                 depth: int = 3
                 ):
        super().__init__()
        self._hidden_size = hidden_size
        self._latent_size = latent_size

        self._flatten_layer = tf.keras.layers.Flatten()
        self._dense_layers = [tf.keras.layers.Dense(units=self._hidden_size
                                                    , activation='relu'
                                                    , kernel_initializer='glorot_normal')
                              for _ in range(depth)]
        self._mean_output = tf.keras.layers.Dense(self._latent_size)
        self._std_output = tf.keras.layers.Dense(self._latent_size)

    def call(self,
             x: tf.Tensor
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self._flatten_layer(x)
        for i in range(len(self._dense_layers)):
            x = self._dense_layers[i](x)

        mean = self._mean_output(x)
        std = self._std_output(x)

        return mean, std


