import numpy as np
import nibabel as nib
import tensorflow as tf
import keras
from keras import layers, Sequential, Layer, Model

# TODO (general): Add typing suggestions to methods where needed/advised/possible
# TODO (general): Add docstrings to all functions and mthods
class ReflectionPadding1D(Layer):
    def __init__(self, padding: int = 1):
        super(ReflectionPadding1D, self).__init__()
        self.padding = padding

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.padding, self.padding], [0, 0]],
                      mode='REFLECT')


def pre_pad(layer: Layer):
    return Sequential([ReflectionPadding1D(padding=1), layer])


class Encoder(Layer):
    def __init__(self, latent_space_dims=32, kernel_size=3):
        super(Encoder, self).__init__()

        # TODO: Add comments to the architecture of the model
        self.latent_space_dims = latent_space_dims
        self.kernel_size = kernel_size

        self.encod_conv1 = pre_pad(
            layers.Conv1D(32, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv1")
        )
        self.encod_conv2 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv2")
        )
        self.encod_conv3 = pre_pad(
            layers.Conv1D(128, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv3")
        )
        self.encod_conv4 = pre_pad(
            layers.Conv1D(256, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv4")
        )
        self.encod_conv5 = pre_pad(
            layers.Conv1D(512, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv5")
        )
        self.encod_conv6 = pre_pad(
            layers.Conv1D(1024, self.kernel_size, strides=1, padding='valid',
                          name="encoder_conv6")
        )

        self.flatten = layers.Flatten(name='flatten')
        self.fc1 = layers.Dense(self.latent_space_dims, name='fc1')

    def call(self, input_data):
        x = input_data

        h1 = tf.nn.relu(self.encod_conv1(x))
        h2 = tf.nn.relu(self.encod_conv2(h1))
        h3 = tf.nn.relu(self.encod_conv3(h2))
        h4 = tf.nn.relu(self.encod_conv4(h3))
        h5 = tf.nn.relu(self.encod_conv5(h4))
        h6 = self.encod_conv6(h5)

        self.encoder_out_size = h6.shape[1:]

        # Flatten
        h7 = self.flatten(h6)
        fc1 = self.fc1(h7)

        return fc1

class Decoder(Layer):
    def __init__(self, encoder_out_size, kernel_size=3):
        super(Decoder, self).__init__()
        self.kernel_size = kernel_size
        self.encoder_out_size = encoder_out_size

        self.fc2 = layers.Dense(8192, name="fc2")
        # TODO (general): Add comments to the architecture of the model
        self.decod_conv1 = pre_pad(
            layers.Conv1D(512, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv1")
        )
        self.upsampl1 = layers.UpSampling1D(size=2, name="upsampling1")
        self.decod_conv2 = pre_pad(
            layers.Conv1D(256, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv2")
        )
        self.upsampl2 = layers.UpSampling1D(size=2, name="upsampling2")
        self.decod_conv3 = pre_pad(
            layers.Conv1D(128, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv3")
        )
        self.upsampl3 = layers.UpSampling1D(size=2, name="upsampling3")
        self.decod_conv4 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv4")
        )
        self.upsampl4 = layers.UpSampling1D(size=2, name="upsampling4")
        self.decod_conv5 = pre_pad(
            layers.Conv1D(32, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv5")
        )
        self.upsampl5 = layers.UpSampling1D(size=2, name="upsampling5")
        self.decod_conv6 = pre_pad(
            layers.Conv1D(3, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv6")
        )

    def call(self, input_data):
        z = input_data
        fc = self.fc2(z)

        # Reshape to match encoder output size
        fc_reshape = tf.reshape(fc, (-1, self.encoder_out_size[0],
                                     self.encoder_out_size[1]))

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


def init_model(latent_space_dims=32, kernel_size=3):
    input_data = keras.Input(shape=(256, 3), name='input_streamline')

    # encode
    encoder = Encoder(latent_space_dims=latent_space_dims,
                      kernel_size=kernel_size)
    encoded = encoder(input_data)

    # decode
    decoder = Decoder(encoder.encoder_out_size,
                      kernel_size=kernel_size)
    decoded = decoder(encoded)
    output_data = decoded

    # Instantiate model and name it
    model = Model(input_data, output_data)
    model.name = 'IncrFeatStridedConvFCUpsampReflectPadAE'
    return model


class IncrFeatStridedConvFCUpsampReflectPadAE():
    # TODO: Complete docstring
    """Strided convolution-upsampling-based AE using reflection-padding and
    increasing feature maps in decoder.
    """

    def __init__(self, latent_space_dims=32, kernel_size=3):

        # Parameter Initialization
        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims
        self.input = keras.Input(shape=(256, 3), name='input_streamline')

        self.model = init_model(latent_space_dims=self.latent_space_dims,
                                kernel_size=self.kernel_size)

    def __call__(self, x):
        return self.model(x)

    def compile(self, optimizer="adam", loss="mean_squared_error",
                metrics=None, loss_weights=None):
        """Configure the model for training
        # TODO: Complete docstring
        Args:
            optimizer (str, optional): _description_. Defaults to "adam".
            loss (_type_, optional): _description_. Defaults to None.
            metrics (_type_, optional): _description_. Defaults to None.
            loss_weights (_type_, optional): _description_. Defaults to None.
        """
        if optimizer is None:
            optimizer = self.optimizer
        if loss is None:
            loss = self.loss
        # Compile the model
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics,
                           loss_weights=loss_weights)

    def summary(self):
        """Get the summary of the model.
        # TODO: Complete docstring
        The summary is textual and includes information about:
        The layers and their order in the model.
        The output shape of each layer.
        """
        return self.model.summary()

    def fit(self, x, y, batch_size=None, epochs=1):
        """_summary_
        # TODO: Complete docstring
        Args:
            x (_type_): _description_
            y (_type_): _description_
            batch_size (_type_, optional): _description_. Defaults to None.
            epochs (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        self.compile()
        if isinstance(x, nib.streamlines.ArraySequence):
            x = np.array(x)
        if isinstance(y, nib.streamlines.ArraySequence):
            y = np.array(y)
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs)
        # TODO (perhaps): write train loop manually?

    def save_weights(self, filepath, overwrite=True):
        """_summary_
        # TODO: Complete docstring
        Args:
            filepath (_type_): _description_
            overwrite (bool, optional): _description_. Defaults to True.
        """
        self.model.save_weights(filepath, overwrite=overwrite)
