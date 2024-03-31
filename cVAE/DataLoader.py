import tensorflow as tf
from typing import Union


class DataLoader():
    def __init__(self,
                 mode: str,
                 random_seed: int):

        self._mode = mode
        self._seed = random_seed

    def _load_and_preprocess_image_tf(self,
                                      image_path: tf.Tensor
                                      ) -> tf.Tensor:

        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=1, expand_animations=False)
        image = tf.image.resize(image, [512, 512])

        max_val = tf.reduce_max(image)
        image = tf.cond(max_val > 0, lambda: image / max_val, lambda: image)
        return image

    def load_dataset(self,
                     length: Union[int, str],
                     batch_size: int,
                     return_iterator: bool = True):

        if self._mode in ['continuous', 'binary']:
            folder_path = f"./{self._mode}/"
            if length == 'full':
                image_paths = tf.io.gfile.glob(folder_path + '*.png')
            else:
                image_paths = tf.io.gfile.glob(folder_path + '*.png')[:length]

        elif self._mode == 'mix':
            if length == 'full':
                image_paths = tf.io.gfile.glob("./continuous/*.png")
                image_paths += tf.io.gfile.glob("./binary/*.png")
            else:
                image_paths = tf.io.gfile.glob("./continuous/*.png")[:length]
                image_paths += tf.io.gfile.glob("./binary/*.png")[:length]

        else:
            raise ValueError('Unknown mode')

        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(self._load_and_preprocess_image_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if return_iterator:
            total_size = len(image_paths)
            train_size = int(0.7 * total_size)

            train_ds = ds.take(train_size).shuffle(10 * batch_size, seed=self._seed).batch(batch_size).prefetch(
                5).repeat()
            validation_ds = ds.skip(train_size).shuffle(10 * batch_size, seed=self._seed).batch(batch_size).prefetch(
                5).repeat()

            return iter(train_ds.as_numpy_iterator()), iter(validation_ds.as_numpy_iterator())
        else:
            images = ds.batch(batch_size).as_numpy_iterator()
            return next(images)


