import tensorflow as tf
from typing import Tuple, Iterator
import os
import numpy as np
from ContinuousLoss import ContinuousLoss
from Encoder import Encoder
from ContinuousDecoder import ContinuousDecoder


class ContinuousVariationalAutoEncoder:
    def __init__(
            self,
            loss_evaluator: ContinuousLoss,
            optimiser: tf.keras.optimizers,
            hidden_size: int,
            latent_size: int,
            output_shape: tf.Tensor
    ):

        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self._output_shape = output_shape
        self.loss_evaluator = loss_evaluator

        self.optimiser = optimiser

        self.encoder = Encoder(self._hidden_size, self._latent_size)
        self.decoder = ContinuousDecoder(self._hidden_size
                                         , self._output_shape)

    @tf.function
    def _train_step(self,
                    x: tf.Tensor,
                    val_x: tf.Tensor) -> Tuple[float, float]:

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.decoder.trainable_variables)
            tape.watch(self.encoder.trainable_variables)

            mean, std = self.encoder(x)
            var = tf.square(std)
            z = mean + std * tf.random.normal(mean.shape)
            mu, logvar = self.decoder(z)

            loss = self.loss_evaluator.evaluate(x, mean, var, mu, logvar)

        decoder_grad = tape.gradient(loss, self.decoder.trainable_variables)
        encoder_grad = tape.gradient(loss, self.encoder.trainable_variables)

        del tape

        self.optimiser.apply_gradients(zip(encoder_grad
                                           , self.encoder.trainable_variables))
        self.optimiser.apply_gradients(zip(decoder_grad
                                           , self.decoder.trainable_variables))

        val_mean, val_std = self.encoder(val_x)
        val_var = tf.square(val_std)
        val_z = val_mean + val_std * tf.random.normal(val_mean.shape)
        val_mu, val_logvar = self.decoder(val_z)

        val_loss = self.loss_evaluator.evaluate(val_x, val_mean
                                                , val_var, val_mu, val_logvar)

        return loss, val_loss

    def train(self,
              train_data: Iterator,
              validation_data: Iterator,
              train_step: int,
              eval_freq: int = 50
              ):

        for step in range(train_step):
            train_batch = tf.convert_to_tensor(next(train_data), dtype='float32')
            val_batch = tf.convert_to_tensor(next(validation_data), dtype='float32')

            train_loss, val_loss = self._train_step(train_batch
                                                    , val_batch)

            if step % eval_freq == 0:
                print('step: {};  train elbo: {}; validation elbo: {};'.format(step
                                                                               , train_loss, val_loss))

    def encode(self, data, save=True):
        mean, std = self.encoder(data)

        if save is True:
            np.savetxt('./encoded_latent_mean.txt', mean.numpy())
            np.savetxt('./encoded_latent_std.txt', std.numpy())

        return mean.numpy(), std.numpy()

    def reconstruct(self, data, save=True):
        mean, std = self.encoder(data)
        z = mean + std * tf.random.normal(mean.shape)
        mu, logvar = self.decoder(z)

        image = mu.numpy()

        if save is True:
            np.savetxt('./reconstructed_images.txt',
                       image.reshape(image.shape[1]
                                     , image.shape[2]))

        return image

    def save_model(self, save_dir='./model_save'):
        encoder_save_path = os.path.join(save_dir, 'encoder')
        decoder_save_path = os.path.join(save_dir, 'decoder')

        os.makedirs(encoder_save_path, exist_ok=True)
        os.makedirs(decoder_save_path, exist_ok=True)

        self.encoder.save(encoder_save_path)
        self.decoder.save(decoder_save_path)

        print(f'models saved successfully in {save_dir}')

    def load_model(self, load_dir='./model_save'):

        encoder_load_path = os.path.join(load_dir, 'encoder')
        decoder_load_path = os.path.join(load_dir, 'decoder')

        if not os.path.exists(encoder_load_path) or not os.path.exists(decoder_load_path):
            raise FileNotFoundError("directory does not exist.")

        self.encoder = tf.keras.models.load_model(encoder_load_path)
        self.decoder = tf.keras.models.load_model(decoder_load_path)

        print(f'models loaded successfully from {load_dir}')