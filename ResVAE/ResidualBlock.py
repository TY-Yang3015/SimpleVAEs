import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides
                                            , padding='same', activation='relu', kernel_initializer='glorot_normal')
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=1
                                            , padding='same', activation='relu', kernel_initializer='glorot_normal')

    def call(self, input_features):
        x = self.conv1(input_features)
        x = self.conv2(x)
        x += input_features
        return tf.nn.relu(x)
