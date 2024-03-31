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

    def _gaussian_log_likelihood(self,
                                 x: tf.Tensor,
                                 mu: tf.Tensor,
                                 logvar: tf.Tensor) -> tf.Tensor:
        log_likelihood = -0.5 * (tf.math.log(2 * self.PI)
                                 + logvar + tf.square(x - mu)
                                 / tf.exp(logvar))
        return tf.reduce_sum(log_likelihood, axis=(1, 2, 3))

    def _reconstruction_loss(self, x, mu):  # sse
        mu = tf.reshape(mu, x.shape)
        return tf.reduce_sum(tf.square(x - mu), axis=(1, 2, 3))

    def _correlation_penalty(self, mean, var):  # deprecated
        epsilon = tf.random.normal(mean.shape)
        z = mean + tf.sqrt(var) * epsilon
        z_centered = z - tf.reduce_mean(z, axis=1, keepdims=True)

        cov_matrix_batch = tf.matmul(z_centered, z_centered, transpose_b=True) / (
                    tf.cast(tf.shape(z_centered)[1], tf.float32) - 1)

        std_dev_batch = tf.sqrt(tf.linalg.diag_part(cov_matrix_batch))

        corr_matrix_batch = cov_matrix_batch / (std_dev_batch[:, tf.newaxis] * std_dev_batch[tf.newaxis, :])

        Hi = 0.5 * tf.reduce_sum(tf.math.log(2 * self.PI * self.E * var))
        H_mat = 0.5 * tf.math.log((2 * self.PI * self.E) ** mean.shape[1]
                                  * tf.linalg.det(corr_matrix_batch[:]))

        return Hi - H_mat

    @tf.function
    def evaluate(self,
                 x: tf.Tensor,
                 mean: tf.Tensor,
                 var: tf.Tensor,
                 mu: tf.Tensor,
                 logvar: tf.Tensor) -> tf.Tensor:
        return -tf.reduce_mean(self._gaussian_log_likelihood(x, mu, logvar)
                               - self._kl_gaussian(mean, var)) + tf.reduce_mean(self._reconstruction_loss(x, mu))
