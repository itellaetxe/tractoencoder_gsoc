import os
from math import sqrt

import numpy as np
import nibabel as nib
import tensorflow as tf
import tensorflow.keras.ops as ops
import keras
from tensorflow.keras import layers, Layer, Model, initializers

from tractoencoder_gsoc.utils import pre_pad
from tractoencoder_gsoc.utils import dict_kernel_size_flatten_encoder_shape


# TODO (general): Add typing suggestions to methods where needed/advised/possible
# TODO (general): Add docstrings to all functions and mthods

def safe_exp(x):
    # Safe exp operation to prevent exp from producing inf values
    return tf.clip_by_value(tf.exp(x), -1e10, 1e10)

class ReparametrizationTrickSampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, **kwargs):
        super(ReparametrizationTrickSampling, self).__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(2208)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]

        # Reparametrization trick (z = z_mean + std_deviation * epsilon)
        epsilon = tf.keras.random.normal(shape=(batch, dim),
                                         seed=self.seed_generator,
                                         stddev=1)
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
        self.encod_batchnorm1 = layers.BatchNormalization(name="encoder_batchnorm1")
        self.encod_conv2 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv2",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_batchnorm2 = layers.BatchNormalization(name="encoder_batchnorm2")
        self.encod_conv3 = pre_pad(
            layers.Conv1D(128, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv3",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_batchnorm3 = layers.BatchNormalization(name="encoder_batchnorm3")
        self.encod_conv4 = pre_pad(
            layers.Conv1D(256, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv4",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_batchnorm4 = layers.BatchNormalization(name="encoder_batchnorm4")
        self.encod_conv5 = pre_pad(
            layers.Conv1D(512, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv5",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_batchnorm5 = layers.BatchNormalization(name="encoder_batchnorm5")
        self.encod_conv6 = pre_pad(
            layers.Conv1D(1024, self.kernel_size, strides=1, padding='valid',
                          name="encoder_conv6",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_batchnorm6 = layers.BatchNormalization(name="encoder_batchnorm6")

        self.flatten = layers.Flatten(name='flatten')

        # For Dense layers
        # Link: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html (Variables section)

        # Z Weights
        self.k_z_mean_weights_initializer = sqrt(1 / dict_kernel_size_flatten_encoder_shape[self.kernel_size])
        self.z_mean_weights_initializer = initializers.RandomUniform(minval=-self.k_z_mean_weights_initializer,
                                                                     maxval=self.k_z_mean_weights_initializer,
                                                                     seed=2208)

        self.k_z_log_var_weights_initializer = sqrt(1 / dict_kernel_size_flatten_encoder_shape[self.kernel_size])
        self.z_log_var_weights_initializer = initializers.RandomUniform(minval=-self.k_z_log_var_weights_initializer,
                                                                        maxval=self.k_z_log_var_weights_initializer,
                                                                        seed=1102)

        # Z Biases
        self.k_z_mean_biases_initializer = self.k_z_mean_weights_initializer
        self.z_mean_biases_initializer = self.z_mean_weights_initializer

        self.k_z_log_var_biases_initializer = self.k_z_log_var_weights_initializer
        self.z_log_var_biases_initializer = self.z_log_var_weights_initializer

        # Instantiate multilayer perceptron for z mean and z log variance
        self.z_mean = layers.Dense(self.latent_space_dims, name='z_mean',
                                   kernel_initializer=self.z_mean_weights_initializer,
                                   bias_initializer=self.z_mean_biases_initializer)

        self.z_log_var = layers.Dense(self.latent_space_dims, name='z_log_var',
                                      kernel_initializer=self.z_log_var_weights_initializer,
                                      bias_initializer=self.z_log_var_biases_initializer)

        # Regressor (r) weights
        self.k_r_mean_weights_initializer = sqrt(1 / dict_kernel_size_flatten_encoder_shape[self.kernel_size])
        self.r_mean_weights_initializer = initializers.RandomUniform(minval=-self.k_r_mean_weights_initializer,
                                                                     maxval=self.k_r_mean_weights_initializer,
                                                                     seed=2208)

        # Regressor (r) biases
        self.k_r_log_var_biases_initializer = sqrt(1 / dict_kernel_size_flatten_encoder_shape[self.kernel_size])
        self.r_log_var_biases_initializer = initializers.RandomUniform(minval=-self.k_r_log_var_biases_initializer,
                                                                       maxval=self.k_r_log_var_biases_initializer,
                                                                       seed=1102)

        # Instantiate multilayer perceptron for r mean and r log variance
        self.r_mean = layers.Dense(1, name="r_mean",
                                   kernel_initializer=self.r_mean_weights_initializer,
                                   bias_initializer=self.r_mean_weights_initializer)

        self.r_log_var = layers.Dense(1, name="r_log_var",
                                      kernel_initializer=self.r_log_var_biases_initializer,
                                      bias_initializer=self.r_log_var_biases_initializer)

        # Sampling Layer
        self.sampling = ReparametrizationTrickSampling()

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_space_dims": keras.saving.serialize_keras_object(self.latent_space_dims),
            "kernel_size": keras.saving.serialize_keras_object(self.kernel_size),
            "encoder_out_size": keras.saving.serialize_keras_object(self.encoder_out_size)
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
        h1 = self.encod_batchnorm1(h1)
        h2 = tf.nn.relu(self.encod_conv2(h1))
        h2 = self.encod_batchnorm2(h2)
        h3 = tf.nn.relu(self.encod_conv3(h2))
        h3 = self.encod_batchnorm3(h3)
        h4 = tf.nn.relu(self.encod_conv4(h3))
        h4 = self.encod_batchnorm4(h4)
        h5 = tf.nn.relu(self.encod_conv5(h4))
        h5 = self.encod_batchnorm5(h5)
        h6 = self.encod_conv6(h5)
        h6 = self.encod_batchnorm6(h6)

        self.encoder_out_size = h6.shape[1:]

        # Flatten
        # First transpose the tensor to match the PyTorch implementation so the flattening is equal
        h7 = tf.transpose(h6, perm=[0, 2, 1])
        h7 = self.flatten(h7)

        # Get the distribution z mean and z log variancee
        z_mean = self.z_mean(h7)
        z_log_var = self.z_log_var(h7)
        z = self.sampling([z_mean, z_log_var])

        r_mean = self.r_mean(h7)
        r_log_var = self.r_log_var(h7)
        r = self.sampling([r_mean, r_log_var])

        return (z_mean, z_log_var, z, r_mean, r_log_var, r)


class Generator(Layer):
    """
    Latent generator based on https://github.com/QingyuZhao/VAE-for-Regression/blob/master/3D_MRI_VAE_regression.py
    """
    def __init__(self, latent_space_dims=32, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.latent_space_dims = latent_space_dims

        self.pz_mean = layers.Dense(self.latent_space_dims,
                                    kernel_constraint=keras.constraints.unit_norm(),
                                    name="pz_mean")
        self.pz_log_var = layers.Dense(1,
                                       kernel_constraint=keras.constraints.max_norm(0),
                                       name="pz_log_var")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_space_dims": keras.saving.serialize_keras_object(self.latent_space_dims)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        latent_space_dims = keras.saving.deserialize_keras_object(config.pop('latent_space_dims'))
        return cls(latent_space_dims, **config)

    def call(self, input_data):
        x = input_data

        pz_mean = self.pz_mean(x)
        pz_log_var = self.pz_log_var(x)

        return (pz_mean, pz_log_var)


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
    input_r = keras.Input(shape=(1,), name='input_r')

    # encode
    encoder = Encoder(latent_space_dims=latent_space_dims,
                      kernel_size=kernel_size)
    # this is the latent vector sampled with the reparametrization trick
    encoder_output = encoder(input_data)
    z_mean, z_log_var, z, r_mean, r_log_var, r = encoder_output

    # generate
    generator = Generator(latent_space_dims=latent_space_dims)
    generator_output = generator(r)
    pz_mean, pz_log_var = generator_output

    # Instantiate encoder model
    model_encoder = Model(input_data,
                          (z_mean, z_log_var, z, r_mean, r_log_var, r, pz_mean, pz_log_var),
                          name="Encoder")

    # decode
    latent_input = keras.Input(shape=(latent_space_dims,), name='z_sampling')
    decoder = Decoder(encoder.encoder_out_size,
                      kernel_size=kernel_size)
    decoded = decoder(latent_input)
    output_data = decoded

    # Instantiate model and name it
    model_decoder = Model(latent_input, output_data, name="Decoder")

    return model_encoder, model_decoder, encoder.encoder_out_size


class IncrFeatStridedConvFCUpsampReflectPadCondVAE(Model):
    # TODO: Complete docstring
    """Strided convolution-upsampling-based VAE using reflection-padding and
    increasing feature maps in decoder.
    """

    def __init__(self, latent_space_dims=32, kernel_size=3,
                 beta: float = 1.0, **kwargs):
        super(IncrFeatStridedConvFCUpsampReflectPadCondVAE, self).__init__(**kwargs)

        # Parameter Initialization
        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims
        self.beta = beta

        self.name = 'IncrFeatStridedConvFCUpsampReflectPadCondVAE'

        # Instantiation
        self.encoder, self.decoder, self.encoder_out_size = init_model(latent_space_dims=latent_space_dims,
                                                                       kernel_size=kernel_size)

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.label_loss_tracker = tf.keras.metrics.Mean(name="label_loss")

        # Instantiate TensorBoard writer
        self.writer = tf.summary.create_file_writer("./logs")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.label_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            input_data = data[0][0]
            input_r = data[0][1]
            encoder_output = self.encoder(input_data, training=True)
            z_mean, z_log_var, z, r_mean, r_log_var, r, pz_mean, pz_log_var = encoder_output
            reconstruction = self.decoder(z)

            # Compute Losses: Reconstruction, KL Divergence, and Label Loss
            reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(input_data, reconstruction)))

            kl_loss = - 0.5 * (1 + z_log_var - pz_log_var - tf.divide(ops.square(z_mean - pz_mean), safe_exp(pz_log_var)) - tf.divide(safe_exp(z_log_var), safe_exp(pz_log_var)))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))

            label_loss = tf.reduce_mean(tf.divide(0.5 * ops.square(r_mean - input_r), safe_exp(r_log_var)) + 0.5 * r_log_var)

            total_loss = reconstruction_loss + self.beta * kl_loss + label_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.label_loss_tracker.update_state(label_loss)

        # Inside your training loop, after calculating total_loss
        with self.writer.as_default():
            tf.summary.scalar('Total Loss', total_loss, step=self.optimizer.iterations)
            tf.summary.scalar('Label Loss', label_loss, step=self.optimizer.iterations)
            tf.summary.scalar('Reconstruction Loss', reconstruction_loss, step=self.optimizer.iterations)
            tf.summary.scalar('KL Loss', kl_loss, step=self.optimizer.iterations)
            self.writer.flush()
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "label_loss": self.label_loss_tracker.result()
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
        z_mean, z_log_var, z, r_mean, r_log_var, r, pz_mean, pz_log_var = self.encoder(x)
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
        # For fitting the model we input a list like x=[streamlines, streamline_lengths]
        if isinstance(kwargs['x'][0], nib.streamlines.ArraySequence):
            kwargs['x'][0] = np.array(kwargs['x'][0])

        return super().fit(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        super().save_weights(*args, **kwargs)

    def save(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        super().save(*args, **kwargs)
