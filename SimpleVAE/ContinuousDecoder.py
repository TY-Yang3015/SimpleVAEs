import tensorflow as tf
from typing import Tuple


class ContinuousDecoder(tf.keras.Model):
    def __init__(self,
                 hidden_size: int,
                 output_shape: tf.Tensor,
                 depth: int = 3
                 ):
        super().__init__()
        self._hidden_size = hidden_size
        self._output_shape = output_shape

        self._dense_layers = [tf.keras.layers.Dense(units=self._hidden_size
                                                    , activation='relu'
                                                    , kernel_initializer='glorot_normal')
                              for _ in range(depth)]
        self._mu_output = tf.keras.layers.Dense(tf.reduce_prod(self._output_shape))
        self._logvar_output = tf.keras.layers.Dense(tf.reduce_prod(self._output_shape))

    def call(self,
             z: tf.Tensor
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        for i in range(len(self._dense_layers)):
            z = self._dense_layers[i](z)

        mu = self._mu_output(z)
        mu = tf.reshape(mu, (mu.shape[0]
                             , self._output_shape[0], self._output_shape[1], self._output_shape[2]))

        logvar = self._logvar_output(z)
        logvar = tf.reshape(logvar, (logvar.shape[0]
                                     , self._output_shape[0], self._output_shape[1], self._output_shape[2]))

        return mu, logvar
