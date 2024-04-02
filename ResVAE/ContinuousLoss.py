import tensorflow as tf
import numpy as np

class ContinuousLoss:
    def __init__(self):
        self.PI = tf.constant(np.pi, dtype='float32')
        self.E = tf.constant(np.e, dtype='float32')

    def _kl_gaussian(self,
                     mean: tf.Tensor,
                     var: tf.Tensor
                     ) -> tf.Tensor:
        return 0.5 * tf.reduce_sum(-tf.math.log(var) - 1.0
                                   + var + tf.square(mean), axis=-1)

    def _reconstruction_loss(self, x, rec):  # mse
        mu = tf.reshape(rec, x.shape)
        return tf.reduce_mean(tf.square(x - rec))

    @tf.function
    def evaluate(self,
                 x: tf.Tensor,
                 mean: tf.Tensor,
                 var: tf.Tensor,
                 rec:tf.Tensor
                ) -> tf.Tensor:

        return tf.abs(tf.reduce_mean(self._kl_gaussian(mean, var))) + 100.*self._reconstruction_loss(x, rec)
        # adjust the 100 (beta param) if needed