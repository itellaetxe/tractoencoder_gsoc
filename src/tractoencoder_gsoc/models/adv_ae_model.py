from math import sqrt

import numpy as np
import nibabel as nib
import tensorflow as tf
import keras
from keras import layers, Layer, Model

from tractoencoder_gsoc.utils import pre_pad, cross_entropy
from tractoencoder_gsoc.utils import dict_kernel_size_flatten_encoder_shape
from tractoencoder_gsoc.models import ae_model
from tractoencoder_gsoc.data_loader import DataLoader
from tractoencoder_gsoc.prior import PriorFactory

class Discriminator(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.kernel_size = kernel_size

        self.dense0 = layers.Dense(128, name="discriminator_dense0")
        self.conv0 = layers.Conv1D(64, self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   name="discriminator_conv0")
        self.do0 = layers.Dropout(0.3, name="discriminator_dropout0")

        self.conv1 = layers.Conv1D(32, self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   name="discriminator_conv1")
        self.do1 = layers.Dropout(0.3, name="discriminator_dropout1")

        self.flatten = layers.Flatten(name="discriminator_flatten")
        self.dense1 = layers.Dense(16, name="discriminator_dense1")
        self.prediction_logits = layers.Dense(1, name="discriminator_prediction_logits")
        self.prediction = tf.math.sigmoid
        self.name = "Discriminator"

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
        x = self.dense0(x)
        x = self.conv0(x)
        x = layers.LeakyReLU()(x)
        x = self.do0(x)

        x = self.conv1(x)
        x = layers.LeakyReLU()(x)
        x = self.do1(x)

        x = self.flatten(x)
        x = self.dense1(x)
        prediction_logits = self.prediction_logits(x)
        prediction = self.prediction(prediction_logits)

        return prediction, prediction_logits


def init_model(latent_space_dims=32, kernel_size=3, n_classes=7):
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
    discriminator_input = layers.Input(shape=(latent_space_dims + n_classes,),
                                       name='discriminator_input')
    decision = discriminator(discriminator_input)
    discriminator_model = Model(discriminator_input, decision)

    return encoder_model, decoder_model, discriminator_model


class JH_Adv_AE(Model):
    def __init__(self, latent_space_dims=32,
                 kernel_size=3,
                 n_classes=7,
                 **kwargs):
        super(JH_Adv_AE, self).__init__(**kwargs)

        # Parameter Initialization
        self.latent_space_dims = latent_space_dims
        self.kernel_size = kernel_size
        self.encoder_out_size = dict_kernel_size_flatten_encoder_shape[kernel_size]
        self.n_classes = n_classes

        self.encoder, self.decoder, self.discriminator = init_model(latent_space_dims=self.latent_space_dims,
                                                                    kernel_size=self.kernel_size)

        self.name = "JH_Adv_AE"

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
