import tensorflow as tf
from DataLoader import DataLoader
from ContinuousLoss import ContinuousLoss
from ConvContinuousVariationalAutoEncoder import ConvContinuousVariationalAutoEncoder
import numpy as np

BATCH_SIZE = 20
TRAIN_STEP = 10000
EVAL_FREQ = 50
RANDOM_SEED = 42
IMAGE_SHAPE = tf.convert_to_tensor([128, 128, 1], dtype='int32')

dl = DataLoader('mix', random_seed=RANDOM_SEED)
train_ds, val_ds = dl.load_dataset(length='full', batch_size=BATCH_SIZE)

decays = [2000, 3000, 4000, 5000]
rates =list(np.array([5e-4, 1e-4, 1e-4, 5e-5, 1e-5]))

learning_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(decays, rates)


optimiser = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
loss = ContinuousLoss()
cVAE = ConvContinuousVariationalAutoEncoder(loss_evaluator=loss,
                                       optimiser_learning_schedule=learning_schedule,
                                       latent_size=8000,
                                       output_shape=IMAGE_SHAPE
                                        )
cVAE.load_model('./ResVAE/mix')
cVAE.train(train_ds,
           val_ds,
           train_step=TRAIN_STEP,
           eval_freq=EVAL_FREQ,
          threshold_loss = [0.1, 0.1]
          )

