import tensorflow as tf
from DataLoader import DataLoader
from ContinuousLoss import ContinuousLoss
from ContinuousVariationalAutoEncoder import ContinuousVariationalAutoEncoder


BATCH_SIZE = 20
TRAIN_STEP = 10000
EVAL_FREQ = 10
RANDOM_SEED = 42
IMAGE_SHAPE = tf.convert_to_tensor([512, 512, 1], dtype='int32')


dl = DataLoader('mix', random_seed=RANDOM_SEED)
train_ds, val_ds = dl.load_dataset(length='full', batch_size=BATCH_SIZE)


decays = [2000, 3000, 4000, 5000]
rates = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
learning_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(decays, rates)


optimiser = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
loss = ContinuousLoss()
cVAE = ContinuousVariationalAutoEncoder(loss_evaluator=loss,
                                        hidden_size=256,
                                       optimiser=optimiser,
                                       latent_size=2000,
                                       output_shape=IMAGE_SHAPE)
cVAE.train(train_ds,
           val_ds,
           train_step=TRAIN_STEP,
           eval_freq=EVAL_FREQ)


