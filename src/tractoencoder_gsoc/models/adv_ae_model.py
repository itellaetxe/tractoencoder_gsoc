from math import sqrt

import numpy as np
import nibabel as nib
import tensorflow as tf
import keras
from keras import layers, Sequential, Layer, Model, initializers

from tractoencoder_gsoc.utils import pre_pad, cross_entropy
from tractoencoder_gsoc.utils import dict_kernel_size_flatten_encoder_shape
from tractoencoder_gsoc.models import ae_model

class Discriminator(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.kernel_size = kernel_size

        self.conv0 = layers.Conv1D(64, self.kernel_size, strides=1,
                                   padding='same', activation=None)
        self.do0 = layers.Dropout(0.3)

        self.conv1 = layers.Conv1D(128, self.kernel_size, strides=1,
                                   padding='same', activation=None)
        self.do1 = layers.Dropout(0.3)

        self.flatten = layers.Flatten()
        self.dense0 = layers.Dense(128)
        self.prediction_logits = layers.Dense(1)
        self.prediction = tf.math.sigmoid

    def get_config(self):
        base_config = super().get_config()
        config = {
            "kernel_size": keras.saving.serialize_keras_object(self.kernel_size)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        kernel_size = keras.saving.deserialize_keras_object(config.pop('kernel_size'))
        return cls(kernel_size, **config)

    def call(self, input_data):
        x = tf.expand_dims(input_data, axis=-1)
        x = self.conv0(x)
        x = layers.LeakyReLU()(x)
        x = self.do0(x)

        x = self.conv1(x)
        x = layers.LeakyReLU()(x)
        x = self.do1(x)

        x = self.flatten(x)
        x = self.dense0(x)
        prediction_logits = self.prediction_logits(x)
        prediction = self.prediction(prediction_logits)

        return prediction, prediction_logits


def init_model(latent_space_dims=32, kernel_size=3):
    input_data = keras.Input(shape=(256, 3), name='input_streamlines')

    # encoder
    encoder = ae_model.Encoder(latent_space_dims=latent_space_dims,
                               kernel_size=kernel_size)
    encoded = encoder(input_data)
    encoder_model = Model(input_data, encoded)

    # decoder
    decoder = ae_model.Decoder(kernel_size=kernel_size,
                               encoder_out_size=encoder.encoder_out_size)
    decoded = decoder(encoded)
    decoder_model = Model(encoded, decoded)

    # discriminator
    discriminator = Discriminator(kernel_size=kernel_size)
    decision = discriminator(encoded)
    discriminator_model = Model(encoded, decision)

    return encoder_model, decoder_model, discriminator_model


class JH_Adv_AE(Model):
    def __init__(self, latent_space_dims=32,
                 kernel_size=3, **kwargs):
        super(JH_Adv_AE, self).__init__(**kwargs)

        # Parameter Initialization
        self.latent_space_dims = latent_space_dims
        self.kernel_size = kernel_size
        self.encoder_out_size = dict_kernel_size_flatten_encoder_shape[kernel_size]

        self.encoder, self.decoder, self.discriminator = init_model(latent_space_dims=self.latent_space_dims,
                                                                    kernel_size=self.kernel_size)

        self.name = "JH_Adv_AE"

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.G_loss_tracker = tf.keras.metrics.Mean(name="G_loss")
        self.D_loss_tracker = tf.keras.metrics.Mean(name="D_loss")
        self.att_reg_loss_tracker = tf.keras.metrics.Mean(name="att_reg_loss")

        # Instantiate TensorBoard writer
        self.writer = tf.summary.create_file_writer("./logs")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.G_loss_tracker,
                self.D_loss_tracker,
                self.att_reg_loss_tracker]

    def get_config(self):
        base_config = super(JH_Adv_AE).get_config()
        config = {
            "kernel_size": keras.saving.serialize_keras_object(self.kernel_size),
            "latent_space_dims": keras.saving.serialize_keras_object(self.latent_space_dims),
            "encoder_out_size": keras.saving.serialize_keras_object(self.encoder_out_size)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        kernel_size = keras.saving.deserialize_keras_object(config.pop('kernel_size'))
        latent_space_dims = keras.saving.deserialize_keras_object(config.pop('latent_space_dims'))
        encoder_out_size = keras.saving.deserialize_keras_object(config.pop('encoder_out_size'))
        return cls(kernel_size, latent_space_dims,
                   encoder_out_size, **config)

    @tf.function
    def train_step(self, data, y_labels, optimizers_dict,
                   label_sample, real_dist, n_classes):
        # Unpack data
        x_batch = data

        # AE Gradient Tape
        with tf.GradientTape() as ae_tape:
            # Run the inputs through the AE (Encode->Decode)
            reconstruction = self.decoder(self.encoder(x_batch),
                                          training=True)
            # Compute Reconstruction Loss
            reconstruction_loss = tf.reduce_mean(tf.math.squared_difference(x_batch, reconstruction))
        # Compute the gradients of the AE
        ae_trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        ae_grads = ae_tape.gradient(reconstruction_loss, ae_trainable_variables)
        optimizers_dict['ae'].apply_gradients(zip(ae_grads, ae_trainable_variables))

        # Discriminator
        with tf.GradientTape() as d_tape:
            label_sample_one_hot = tf.one_hot(label_sample, n_classes)
            real_dist_label = tf.concat([real_dist, label_sample_one_hot], axis=1)
            # Run the input through the encoder
            fake_dist = self.encoder(x_batch, training=True)
            fake_dist_label = tf.concat([fake_dist, y_labels], axis=1)

            # Run the latent vectors through the discriminator
            _, real_logits = self.discriminator(real_dist_label, training=True)
            _, fake_logits = self.discriminator(fake_dist_label, training=True)
            # Discriminator loss
            loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_logits), logits=real_logits)
            loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_logits), logits=fake_logits)
            D_loss = tf.reduce_mean(loss_real + loss_fake)

        # Compute the gradients of discriminator
        d_grads = d_tape.gradient(D_loss, self.discriminator.trainable_variables)
        optimizers_dict['discriminator'].apply_gradients(zip(d_grads,
                                                             self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            encoder_output = self.encoder(x_batch, training=True)
            encoder_output_label = tf.concat([encoder_output, y_labels], axis=1)
            _, disc_fake_logits = self.discriminator(encoder_output_label, training=True)
            # Generator Loss
            G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(disc_fake_logits), logits=disc_fake_logits))
        # Compute the gradients of generator
        g_grads = g_tape.gradient(G_loss, self.encoder.trainable_variables)
        optimizers_dict['encoder'].apply_gradients(zip(g_grads, self.encoder.trainable_variables))

        total_loss = reconstruction_loss + G_loss + D_loss

        # Update loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.D_loss_tracker.update_state(D_loss)
        self.G_loss_tracker.update_state(G_loss)

        # Inside your training loop, after calculating total_loss
        with self.writer.as_default():
            tf.summary.scalar('Total Loss', total_loss, step=self.optimizer.iterations)
            tf.summary.scalar('G Loss', G_loss, step=self.optimizer.iterations)
            tf.summary.scalar('D Loss', D_loss, step=self.optimizer.iterations)
            self.writer.flush()
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "G_loss": self.G_loss.result(),
            "D_loss": self.D_loss.result()
        }

    def compile(self, **kwargs):
        """
        Configure the model for training
        """
        if 'optimizer' in kwargs:
            self.ae_optimizer, self.G_optimizer, self.D_optimizer = kwargs['optimizer']
            if hasattr(kwargs['optimizer'][0], 'weight_decay'):  # AE optimizer
                kwargs['optimizer'][0].weight_decay = 0.13
            else:
                print("Optimizer does not have a weight_decay attribute. Ignoring...")

        # Call the superclass's compile with the modified kwargs
        super(JH_Adv_AE).compile(**kwargs)

    def call(self, input_data):
        """
        # TODO: Complete docstring
        """
        x = input_data
        z = self.encoder(x)
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
        super(JH_Adv_AE, self).save_weights(*args, **kwargs)

    def save(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        super(JH_Adv_AE, self).save(*args, **kwargs)
