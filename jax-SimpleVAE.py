from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags, logging

from tqdm.auto import tqdm
from PIL import Image
import tensorflow as tf
import os


flags.DEFINE_integer("batch_size", 20, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 20000, "Number of training steps.")
flags.DEFINE_integer("eval_frequency", 10, "How often to evaluate the model.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")
FLAGS = flags.FLAGS


PRNGKey = jnp.ndarray
Batch = Mapping[str, np.ndarray]

IMAGE_SHAPE: Sequence[int] = (512, 512, 1)


def load_dataset(batch_size: int) -> Iterator[Batch]:
    def load_and_preprocess_image(image_path, target_size=(512, 512)):
        image_path = image_path.replace('._', '')
        image = Image.open(image_path)
        image = image.convert('L')
        image = image.resize(target_size)
        if np.max(image) == 0:
            return np.expand_dims(np.array(image), axis=-1)
        image_array = np.array(image) / np.max(image)
        image_array = np.expand_dims(image_array, axis=-1)
        return image_array

    folder_path = "./continuous/"
    image_paths = os.listdir(folder_path)[:100]

    images = [load_and_preprocess_image(os.path.join(folder_path, img_path)) for img_path in tqdm(image_paths)]

    ds = tf.data.Dataset.from_tensor_slices(images)

    total_size = len(images)
    train_size = int(0.7 * total_size)

    train_ds = ds.take(train_size)
    validation_ds = ds.skip(train_size)

    train_ds = train_ds.shuffle(buffer_size=10 * batch_size, seed=FLAGS.random_seed)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=5)
    train_ds = train_ds.repeat()

    validation_ds = validation_ds.shuffle(buffer_size=10 * batch_size, seed=FLAGS.random_seed)
    validation_ds = validation_ds.batch(batch_size)
    validation_ds = validation_ds.prefetch(buffer_size=5)
    validation_ds = validation_ds.repeat()

    return iter(train_ds.as_numpy_iterator()), iter(validation_ds.as_numpy_iterator())


class Encoder(hk.Module):
    """Encoder model."""

    def __init__(self,
                 hidden_size: int,
                 latent_size: int
                 ):
        super().__init__()
        self._hidden_size = hidden_size
        self._latent_size = latent_size

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = hk.Flatten()(x)
        x = hk.Linear(self._hidden_size)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._hidden_size)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._hidden_size)(x)
        x = jax.nn.relu(x)

        mean = hk.Linear(self._latent_size)(x)
        log_stddev = hk.Linear(self._latent_size)(x)
        stddev = jnp.exp(log_stddev)

        return mean, stddev




class BinaryDecoder(hk.Module):
    """Decoder model."""

    def __init__(
        self,
        hidden_size: int,
        output_shape: Sequence[int],
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._output_shape = output_shape

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = hk.Linear(self._hidden_size)(z)
        z = jax.nn.relu(z)

        logits = hk.Linear(np.prod(self._output_shape))(z)
        logits = jnp.reshape(logits, (-1, *self._output_shape))

        return logits


class ContinuousDecoder(hk.Module):
    def __init__(self, hidden_size: int, output_shape: Sequence[int]):
        super().__init__()
        self._hidden_size = hidden_size
        self._output_shape = output_shape

    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z = hk.Linear(self._hidden_size)(z)
        z = jax.nn.relu(z)

        mu = hk.Linear(np.prod(self._output_shape))(z)
        mu = jnp.reshape(mu, (-1, *self._output_shape))

        log_var = hk.Linear(np.prod(self._output_shape))(z)
        log_var = jnp.reshape(log_var, (-1, *self._output_shape))

        return mu, log_var


class VAEOutputBinary(NamedTuple):
    image: jnp.ndarray
    mean: jnp.ndarray
    stddev: jnp.ndarray
    logits: jnp.ndarray

class VAEOutputContinuous(NamedTuple):
    image: jnp.ndarray
    mean: jnp.ndarray
    stddev: jnp.ndarray
    mu: jnp.ndarray
    log_var: jnp.ndarray


class VariationalAutoEncoder(hk.Module):

    def __init__(
        self,
        mode: str,
        hidden_size: int = 2000,
        latent_size: int = 128,
        output_shape: Sequence[int] = IMAGE_SHAPE,
    ):
        super().__init__()
        self._mode = mode
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self._output_shape = output_shape

    def __call__(self, x: jnp.ndarray):
        x = x.astype(jnp.float32)

        if self._mode == "continuous":
            mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
            z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
            mu, log_var = ContinuousDecoder(self._hidden_size, self._output_shape)(z)
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(hk.next_rng_key(), log_var.shape)
            image = mu + eps * std
            return VAEOutputContinuous(image, mean, stddev, mu, log_var)

        if self._mode == "convolutional":
            mean, stddev = ConvEncoder(self._latent_size)(x)
            z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
            mu, log_var = ConvDecoder(self._output_shape)(z)
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(hk.next_rng_key(), log_var.shape)
            image = mu + eps * std
            return VAEOutputContinuous(image, mean, stddev, mu, log_var)

        elif self._mode == "binary":
            mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
            z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
            logits = BinaryDecoder(self._hidden_size, self._output_shape)(z)
            p = jax.nn.sigmoid(logits)
            image = jax.random.bernoulli(hk.next_rng_key(), p)

            return VAEOutputBinary(image, mean, stddev, logits)
        else:
            raise ValueError("unknown mode.")

    def encode(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode input into latent space representation."""
        x = x.astype(jnp.float32)
        mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
        return mean, stddev


def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    if x.shape != logits.shape:
        raise ValueError("inputs x and logits must be of the same shape")

    x = jnp.reshape(x, (x.shape[0], -1))
    logits = jnp.reshape(logits, (logits.shape[0], -1))

    return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)


def mse_loss(x: jnp.ndarray, reconstruction: jnp.ndarray) -> jnp.ndarray:
    if x.shape != reconstruction.shape:
        raise ValueError("inputs x and reconstruction must be of the same shape")
    return jnp.mean((x - reconstruction) ** 2)


def main(_):
    FLAGS.alsologtostderr = True
    mode = "continuous"

    model = hk.transform(
        lambda x: VariationalAutoEncoder(mode)(x)
    )  


    #optimizer = optax.sgd(FLAGS.learning_rate)
    schedule = optax.exponential_decay(init_value=FLAGS.learning_rate,
                                       transition_begin=100,
                                       decay_rate=0.01,
                                       transition_steps=FLAGS.training_steps)

    optimizer = optax.adam(learning_rate=schedule)


    @jax.jit
    def loss_fn_binary(
        params: hk.Params,
        rng_key: PRNGKey,
        batch: Batch,
    ) -> jnp.ndarray:
        outputs = model.apply(params, rng_key, batch)

        #log_likelihood = -binary_cross_entropy(batch, outputs.logits)

        log_likelihood = mse_loss(batch, outputs.image)


        kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
        elbo = log_likelihood - kl

        return -jnp.mean(elbo)

    def gaussian_log_likelihood(x, mu, log_var):
        """Compute Gaussian log likelihood."""
        log_likelihood = -0.5 * (jnp.log(2 * jnp.pi) + log_var + jnp.square(x - mu) / jnp.exp(log_var))
        return jnp.sum(log_likelihood, axis=(1, 2, 3))

    @jax.jit
    def loss_fn_continuous(params: hk.Params, rng_key: PRNGKey, batch: Batch) -> jnp.ndarray:
        """Computes the negative ELBO for a batch."""
        outputs = model.apply(params, rng_key, batch)
        log_likelihood = gaussian_log_likelihood(batch, outputs.mu, outputs.log_var)
        kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
        elbo = log_likelihood - kl
        return -jnp.mean(elbo)

    @jax.jit
    def update(
        params: hk.Params,
        rng_key: PRNGKey,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:
        grads = jax.grad(loss_fn_continuous)(params, rng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    rng_seq = hk.PRNGSequence(FLAGS.random_seed)
    params = model.init(next(rng_seq), np.zeros((1, *IMAGE_SHAPE)))
    opt_state = optimizer.init(params)

    train_ds, valid_ds = load_dataset(FLAGS.batch_size)

    for step in range(FLAGS.training_steps):
        params, opt_state = update(
            params,
            next(rng_seq),
            opt_state,
            next(train_ds),
        )

        if step % FLAGS.eval_frequency == 0:
            val_loss = loss_fn_continuous(params, next(rng_seq), next(valid_ds))
            logging.info("STEP: %5d; Validation ELBO: %.3f", step, -val_loss)

        if step+1 >= FLAGS.training_steps:
            logging.info("training ended.")

            return 0


app.run(main)