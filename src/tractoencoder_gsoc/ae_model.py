import tensorflow as tf
from tensorflow.keras import layers


class ReflectionPadding1D(tf.keras.layers.Layer):
    def __init__(self, padding):
        super(ReflectionPadding1D, self).__init__()
        self.padding = padding

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.padding, self.padding], [0, 0]], mode='REFLECT')


class IncrFeatStridedConvFCUpsampReflectPadAE(tf.keras.Model):
    """Strided convolution-upsampling-based AE using reflection-padding and
    increasing feature maps in decoder.
    """

    def __init__(self, latent_space_dims):
        super(IncrFeatStridedConvFCUpsampReflectPadAE, self).__init__()

        self.kernel_size = 3
        self.latent_space_dims = latent_space_dims

        self.pad = ReflectionPadding1D(1)

        def pre_pad(layer: tf.keras.layers.Layer):
            return tf.keras.Sequential([self.pad, layer])

        self.encod_conv1 = pre_pad(
            layers.Conv1D(32, self.kernel_size, strides=2, padding='valid')
        )
        self.encod_conv2 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=2, padding='valid')
        )
        self.encod_conv3 = pre_pad(
            layers.Conv1D(128, self.kernel_size, strides=2, padding='valid')
        )
        self.encod_conv4 = pre_pad(
            layers.Conv1D(256, self.kernel_size, strides=2, padding='valid')
        )
        self.encod_conv5 = pre_pad(
            layers.Conv1D(512, self.kernel_size, strides=2, padding='valid')
        )
        self.encod_conv6 = pre_pad(
            layers.Conv1D(1024, self.kernel_size, strides=1, padding='valid')
        )

        self.fc1 = layers.Dense(self.latent_space_dims)
        self.fc2 = layers.Dense(8192)

        self.decod_conv1 = pre_pad(
            layers.Conv1D(512, self.kernel_size, strides=1, padding='valid')
        )
        self.upsampl1 = layers.UpSampling1D(size=2)
        self.decod_conv2 = pre_pad(
            layers.Conv1D(256, self.kernel_size, strides=1, padding='valid')
        )
        self.upsampl2 = layers.UpSampling1D(size=2)
        self.decod_conv3 = pre_pad(
            layers.Conv1D(128, self.kernel_size, strides=1, padding='valid')
        )
        self.upsampl3 = layers.UpSampling1D(size=2)
        self.decod_conv4 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=1, padding='valid')
        )
        self.upsampl4 = layers.UpSampling1D(size=2)
        self.decod_conv5 = pre_pad(
            layers.Conv1D(32, self.kernel_size, strides=1, padding='valid')
        )
        self.upsampl5 = layers.UpSampling1D(size=2)
        self.decod_conv6 = pre_pad(
            layers.Conv1D(3, self.kernel_size, strides=1, padding='valid')
        )

    def encode(self, x):
        h1 = tf.nn.relu(self.encod_conv1(x))
        h2 = tf.nn.relu(self.encod_conv2(h1))
        h3 = tf.nn.relu(self.encod_conv3(h2))
        h4 = tf.nn.relu(self.encod_conv4(h3))
        h5 = tf.nn.relu(self.encod_conv5(h4))
        h6 = self.encod_conv6(h5)

        self.encoder_out_size = tf.shape(h6)[1:]

        # Flatten
        h7 = tf.reshape(h6, (-1, self.encoder_out_size[0] * self.encoder_out_size[1]))

        fc1 = self.fc1(h7)

        return fc1

    def decode(self, z):
        fc = self.fc2(z)
        fc_reshape = tf.reshape(fc, (-1, self.encoder_out_size[0], self.encoder_out_size[1]))
        h1 = tf.nn.relu(self.decod_conv1(fc_reshape))
        h2 = self.upsampl1(h1)
        h3 = tf.nn.relu(self.decod_conv2(h2))
        h4 = self.upsampl2(h3)
        h5 = tf.nn.relu(self.decod_conv3(h4))
        h6 = self.upsampl3(h5)
        h7 = tf.nn.relu(self.decod_conv4(h6))
        h8 = self.upsampl4(h7)
        h9 = tf.nn.relu(self.decod_conv5(h8))
        h10 = self.upsampl5(h9)
        h11 = self.decod_conv6(h10)

        return h11

    def call(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


if __name__ == '__main__':
    # Example of using the model
    latent_space_dims = 128
    model = IncrFeatStridedConvFCUpsampReflectPadAE(latent_space_dims)

    # Assuming input shape is (batch_size, sequence_length, num_features)
    input_shape = (1, 256, 3)  # Example input shape
    dummy_input = tf.random.normal(input_shape)
    model.call(dummy_input)
    model.summary()
