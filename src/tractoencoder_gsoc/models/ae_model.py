from math import sqrt
import numpy as np
import nibabel as nib
import tensorflow as tf
import keras
from keras import layers, Sequential, Layer, Model, initializers
from tractoencoder_gsoc.utils import pre_pad
from tractoencoder_gsoc.utils import dict_kernel_size_flatten_encoder_shape


# TODO (general): Add typing suggestions to methods where needed/advised/possible
# TODO (general): Add docstrings to all functions and mthods


class Encoder(Layer):
    def __init__(self, latent_space_dims=32, kernel_size=3, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        # TODO: Add comments to the architecture of the model
        self.latent_space_dims = latent_space_dims
        self.kernel_size = kernel_size

        # Weight and bias initializers for Conv1D layers (matching PyTorch initialization)
        # Link: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html (Variables section)
        # Weights
        self.k_conv1d_weights_initializer = sqrt(1 / (3 * self.kernel_size))
        self.conv1d_weights_initializer = initializers.RandomUniform(minval=-self.k_conv1d_weights_initializer,
                                                                     maxval=self.k_conv1d_weights_initializer,
                                                                     seed=2208)
        # Biases
        self.k_conv1d_biases_initializer = self.k_conv1d_weights_initializer
        self.conv1d_biases_initializer = self.conv1d_weights_initializer

        self.encod_conv1 = pre_pad(
            layers.Conv1D(32, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv1",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv2 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv2",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv3 = pre_pad(
            layers.Conv1D(128, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv3",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv4 = pre_pad(
            layers.Conv1D(256, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv4",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv5 = pre_pad(
            layers.Conv1D(512, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv5",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv6 = pre_pad(
            layers.Conv1D(1024, self.kernel_size, strides=1, padding='valid',
                          name="encoder_conv6",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )

        self.flatten = layers.Flatten(name='flatten')

        # For Dense layers
        # Link: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html (Variables section)
        # Weights
        self.k_dense_weights_initializer = sqrt(1 / dict_kernel_size_flatten_encoder_shape[self.kernel_size])
        self.dense_weights_initializer = initializers.RandomUniform(minval=-self.k_dense_weights_initializer,
                                                                    maxval=self.k_dense_weights_initializer,
                                                                    seed=2208)
        # Biases
        self.k_dense_biases_initializer = self.k_dense_weights_initializer
        self.dense_biases_initializer = self.dense_weights_initializer

        self.fc1 = layers.Dense(self.latent_space_dims, name='fc1',
                                kernel_initializer=self.dense_weights_initializer,
                                bias_initializer=self.dense_biases_initializer)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_space_dims": keras.saving.serialize_keras_object(self.latent_space_dims),
            "kernel_size": keras.saving.serialize_keras_object(self.kernel_size)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        latent_space_dims = keras.saving.deserialize_keras_object(config.pop('latent_space_dims'))
        kernel_size = keras.saving.deserialize_keras_object(config.pop('kernel_size'))
        return cls(latent_space_dims, kernel_size, **config)

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
        # First transpose the tensor to match the PyTorch implementation so the flattening is equal
        h7 = tf.transpose(h6, perm=[0, 2, 1])
        h7 = self.flatten(h7)
        fc1 = self.fc1(h7)

        return fc1

class Decoder(Layer):
    def __init__(self, encoder_out_size, kernel_size=3, **kwargs):
        super(Decoder, self).__init__(**kwargs)
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

    def get_config(self):
        base_config = super().get_config()
        config = {
            "encoder_out_size": keras.saving.serialize_keras_object(self.encoder_out_size),
            "kernel_size": keras.saving.serialize_keras_object(self.kernel_size)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        encoder_out_size = keras.saving.deserialize_keras_object(config.pop('encoder_out_size'))
        kernel_size = keras.saving.deserialize_keras_object(config.pop('kernel_size'))
        return cls(encoder_out_size, kernel_size, **config)

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

    def compile(self, **kwargs):
        """
        Configure the model for training
        """
        kwargs['optimizer'].weight_decay = 0.13
        self.model.compile(**kwargs)

    def summary(self, **kwargs):
        """
        Get the summary of the model.
        # TODO: Complete docstring
        The summary is textual and includes information about:
        The layers and their order in the model.
        The output shape of each layer.
        """
        return self.model.summary(**kwargs)

    def fit(self, *args, **kwargs,):
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
        if isinstance(kwargs['x'], nib.streamlines.ArraySequence):
            kwargs['x'] = np.array(kwargs['x'])
        if isinstance(kwargs['y'], nib.streamlines.ArraySequence):
            kwargs['y'] = np.array(kwargs['y'])
        return self.model.fit(*args, **kwargs)
        # TODO (perhaps): write train loop manually?

    def save_weights(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        self.model.save_weights(*args, **kwargs)

    def save(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        self.model.save(*args, **kwargs)
