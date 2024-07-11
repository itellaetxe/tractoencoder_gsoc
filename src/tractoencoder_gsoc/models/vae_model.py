import os
from math import sqrt

import numpy as np
import nibabel as nib
import tensorflow as tf
import tensorflow.keras.ops as ops
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Layer, Model, initializers

from tractoencoder_gsoc.utils import pre_pad
from tractoencoder_gsoc.utils import dict_kernel_size_flatten_encoder_shape


# TODO (general): Add typing suggestions to methods where needed/advised/possible
# TODO (general): Add docstrings to all functions and mthods

def safe_exp(x):
    # Safe exp operation to prevent exp from producing inf values
    return tf.clip_by_value(tf.exp(x), -1e20, 1e20)

class ReparametrizationTrickSampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, **kwargs):
        super(ReparametrizationTrickSampling, self).__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(2208)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]

        # Reparametrization trick (z = z_mean + sqrt(var) * epsilon)
        epsilon = keras.random.normal(shape=(batch,
                                             dim),
                                      seed=self.seed_generator)
        return z_mean + safe_exp(0.5 * z_log_var) * epsilon


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
        self.k_z_mean_weights_initializer = sqrt(1 / dict_kernel_size_flatten_encoder_shape[self.kernel_size])
        self.z_mean_weights_initializer = initializers.RandomUniform(minval=-self.k_z_mean_weights_initializer,
                                                                     maxval=self.k_z_mean_weights_initializer,
                                                                     seed=2208)

        self.k_z_log_var_weights_initializer = sqrt(1 / dict_kernel_size_flatten_encoder_shape[self.kernel_size])
        self.z_log_var_weights_initializer = initializers.RandomUniform(minval=-self.k_z_log_var_weights_initializer,
                                                                        maxval=self.k_z_log_var_weights_initializer,
                                                                        seed=1102)

        # Biases
        self.k_z_mean_biases_initializer = self.k_z_mean_weights_initializer
        self.z_mean_biases_initializer = self.z_mean_weights_initializer

        self.k_z_log_var_biases_initializer = self.k_z_log_var_weights_initializer
        self.z_log_var_biases_initializer = self.z_log_var_weights_initializer

        # Instantiate multilayer perceptron for mean and log variance
        self.z_mean = layers.Dense(self.latent_space_dims, name='z_mean',
                                   kernel_initializer=self.z_mean_weights_initializer,
                                   bias_initializer=self.z_mean_biases_initializer)

        self.z_log_var = layers.Dense(self.latent_space_dims, name='z_log_bar',
                                      kernel_initializer=self.z_log_var_weights_initializer,
                                      bias_initializer=self.z_log_var_biases_initializer)

        # Sampling Layer
        self.sampling = ReparametrizationTrickSampling()

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

        # Get the distribution mean and log variancee
        z_mean = self.z_mean(h7)
        z_log_var = self.z_log_var(h7)
        z = self.sampling([z_mean, z_log_var])

        return (z_mean, z_log_var, z)


class Decoder(Layer):
    def __init__(self, encoder_out_size,
                 kernel_size=3,
                 latent_space_dims=32,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims
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
        # z: latent vector sampled from z_mean and z_log_var using the
        # reparametrization trick
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
    # this is the latent vector sampled with the reparametrization trick
    encoder_output = encoder(input_data)
    z_mean, z_log_var, z = encoder_output
    # Instantiate encoder model
    model_encoder = Model(input_data, (z_mean, z_log_var, z), name="Encoder")

    # decode
    latent_input = keras.Input(shape=(latent_space_dims,), name='z_sampling')
    decoder = Decoder(encoder.encoder_out_size,
                      kernel_size=kernel_size)
    decoded = decoder(latent_input)
    output_data = decoded

    # Instantiate model and name it
    model_decoder = Model(latent_input, output_data, name="Decoder")

    return model_encoder, model_decoder


class IncrFeatStridedConvFCUpsampReflectPadVAE(Model):
    # TODO: Complete docstring
    """Strided convolution-upsampling-based VAE using reflection-padding and
    increasing feature maps in decoder.
    """

    def __init__(self, latent_space_dims=32, kernel_size=3, **kwargs):
        super(IncrFeatStridedConvFCUpsampReflectPadVAE, self).__init__(**kwargs)

        # Parameter Initialization
        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims

        self.name = 'IncrFeatStridedConvFCUpsampReflectPadVAE'
        # Instantiation
        self.encoder, self.decoder = init_model(latent_space_dims=latent_space_dims,
                                                kernel_size=kernel_size)

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.kl_weight = tf.Variable(0.005, trainable=False)

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def train_step(self, data):
        epsilon = 1e-7  # To prevent log(0) in kl_loss
        with tf.GradientTape() as tape:
            input_data = data
            z_mean, z_log_var, z = self.encoder(input_data, training=True)
            reconstruction = self.decoder(z)
            reconstruction = tf.clip_by_value(reconstruction, epsilon, 1 - epsilon)  # Clip reconstruction values
            mean_squared_error = tf.keras.losses.mse(input_data,
                                                                    reconstruction)
            reconstruction_loss = ops.mean(
                ops.sum(
                    mean_squared_error,
                    axis=1,
                )
            )
            kl_loss = 0 * (1 + z_log_var - ops.square(z_mean) - safe_exp(0.5 * z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]  # Clip gradients to prevent explosion
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def compile(self, **kwargs):
        """
        Configure the model for training
        """
        if 'optimizer' in kwargs:
            if hasattr(kwargs['optimizer'], 'weight_decay'):
                kwargs['optimizer'].weight_decay = 0.13
            else:
                print("Optimizer does not have a weight_decay attribute. Ignoring...")

        # Call the superclass's compile with the modified kwargs
        super().compile(**kwargs)

    def call(self, input_data):
        """
        # TODO: Complete docstring
        """
        x = input_data
        z_mean, z_log_var, z = self.encoder(x)
        decoded = self.decoder(z)

        return decoded

    def fit(self, *args, **kwargs):
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

        return super().fit(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        self.save_weights(*args, **kwargs)

    def save(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        self.save(*args, **kwargs)
