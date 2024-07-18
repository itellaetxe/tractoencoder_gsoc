# -*- coding: utf-8 -*-

from keras import backend as K
from keras import losses
from keras import Model
from keras.layers import (Conv1D, Dense, Flatten, Input, Lambda, Reshape)

from VITALabAI.model.generative.layer_utils import \
    (Conv1DTranspose, SymmetricPadding1D)
from VITALabAI.model.generative.tractography.tractography_base import \
    TractographyBase


def sample_latent_space(args, epsilon_std=0.01):
    # Reparameterization trick: instead of sampling from q(z|X),
    # sample epsilon = N(0,stddev)
    # z = mu + sqrt(var) * epsilon
    mu, log_sigma = args
    batch_size = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch_size, dim), mean=0.,
                              stddev=epsilon_std)
    z = mu + K.exp(log_sigma / 2) * epsilon
    return z


class TractographyCNNVariationalAutoEncoder(TractographyBase):
    """Convolutional Variational Autoencoder (VAE) model for tractography.
    """

    def __init__(self, dataset, input_shape, num_classes, loss_fn=None,
                 optimizer=None, metrics=None, loss_weights=None,
                 num_blocks=2, latent_space_dim=50, name=None):
        """Initializes the tractography convolutional VAE model.

        Parameters
        ----------
        dataset : dict
            Tractography dataset.
        input_shape : Tuple[int, int]
            Shape of the input data.
        num_classes : int
            Number of classes.
        loss_fn : dict, optional
            Loss functions for training the model.
        optimizer : keras.optimizer, optional
            Instance of a Keras optimizer.
        metrics : list, optional
            Metrics to evaluate the model output.
        loss_weights : dict, optional
            Weights for the loss functions.
        num_blocks : int, optional
            Number of convolutional blocks.
        latent_space_dim : int, optional
            Latent space dimensions.
        name : str, optional
            Name of the model.
        """

        self._dataset = dataset

        # ToDo
        # Fixme: get from dataset
        self._input_shape = input_shape
        self._num_classes = num_classes

        if not num_blocks % 2 == 0:
            raise ValueError('The number of blocks must be even.\n'
                             'Found: {}'.format(num_blocks))

        self._num_blocks = num_blocks
        self._latent_space_dim = latent_space_dim

        self._encoder = None
        self._decoder = None
        self._streamline_predictor = None

        self._num_filters = 32  # Number of convolutional filters to use
        self._kernel_size = 3  # Convolution kernel size
        self._padding = 1  # Padding for symmetric padding

        super().__init__(dataset, loss_fn=loss_fn, optimizer=optimizer,
                         metrics=metrics, loss_weights=loss_weights,
                         name=name)

    @property
    def encoder(self):
        """Get the encoder.
         """

        return self._encoder

    @property
    def decoder(self):
        """Get the decoder.
         """

        return self._decoder

    def build_model(self):
        """Builds the Convolutional Variational Autoencoder model.

        Returns
        -------
        encoder : keras.Model
            Convolutional Variational Autoencoder model.
        """

        self._encoder = self._build_encoder()

        print('Model: %s built.' % self._encoder.name)
        self._encoder.summary()

        self._decoder = self._build_decoder()
        print('Model: %s built.' % self._decoder.name)
        self._decoder.summary()

        # Instantiate VAE model
        input_layer = self._encoder.get_input_at(0)
        decoded_sample = self._decoder(self._encoder(input_layer)[2])

        self.model = Model(inputs=input_layer, outputs=decoded_sample,
                            name='tractography_cnn_vae')
        print('VAE built')
        self.model.summary()

        return self.model

    def _build_encoder(self):
        """Builds the VAE encoder.

        Returns
        -------
        encoder : keras.Model
            Encoder model.
        """

        # Our input shape is (num_points, num_dimensions): num_dimensions
        # can be thought as the channels in an image
        input = Input(shape=(self._input_shape[0], self._input_shape[1],),
                      name='input_streamline')
        x = Reshape((self._input_shape[0], self._input_shape[1]))(input)

        # Fixme
        # The self.num_filters property should not be changed within the
        # class, but the number of filters in the last layer is required by
        # the decoder for reconstruction.
        for i in range(self._num_blocks):
            self._num_filters *= 2
            x = SymmetricPadding1D(padding=self._padding,
                                   input_shape=self._input_shape)(x)
            x = Conv1D(filters=self._num_filters,
                       kernel_size=self._kernel_size, strides=1, padding='valid',
                       activation='relu')(x)
            x = SymmetricPadding1D(padding=self._padding,
                                   input_shape=K.int_shape(x))(x)
            x = Conv1D(filters=self._num_filters,
                       kernel_size=self._kernel_size, strides=2, padding='valid',
                       activation='relu')(x)

        # Shape info needed to build the decoder
        self.last_feat_map_shape = K.int_shape(x)

        # Generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(self._latent_space_dim, activation='relu')(x)

        # Default activations used (linear)
        z_mean = Dense(self._latent_space_dim, name='z_mean')(x)
        z_log_var = Dense(self._latent_space_dim, name='z_log_var')(x)

        # Use reparameterization trick to push the sampling out as input.
        # Note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sample_latent_space,
                   output_shape=(self._latent_space_dim,),
                   name='z')([z_mean, z_log_var])

        encoder = Model(inputs=input, outputs=[z_mean, z_log_var, z],
                        name='encoder')

        return encoder

    def _build_decoder(self):
        """Builds the VAE decoder.

        Returns
        -------
        decoder : keras.Model
            Decoder model.
        """

        encoded_input = Input(shape=(self._latent_space_dim,),
                              name='latent_space_sample')

        # Project latent space to activation maps
        x = Dense(self.last_feat_map_shape[1] * self.last_feat_map_shape[2],
                  activation='relu')(encoded_input)
        x = Reshape((self.last_feat_map_shape[1],
                     self.last_feat_map_shape[2]))(x)

        for i in range(self._num_blocks):
            x = SymmetricPadding1D(padding=self._padding,
                                   input_shape=self._input_shape)(x)
            x = Conv1DTranspose(x, filters=self._num_filters,
                                kernel_size=self._kernel_size, strides=2,
                                padding='valid', activation='relu')
            x = SymmetricPadding1D(padding=self._padding,
                                   input_shape=self._input_shape)(x)
            x = Conv1D(filters=self._num_filters,
                       kernel_size=self._kernel_size, strides=1,
                       padding='valid', activation='relu')(x)
            self._num_filters //= 2

        # Linear activation in output
        output = Conv1D(filters=3, kernel_size=self._kernel_size, strides=1,
                        padding='valid', activation=None,
                        name='decoder_output')(x)

        decoder = Model(inputs=encoded_input, outputs=output, name='decoder')

        return decoder

    def loss_function(self, y_true, y_pred):
        """Computes the VAE loss. The VAE loss is computed as the mean of the
        Kullback-Leibler (KL) divergence loss and the reconstruction loss,
        computed as the mean square error (MSE) between the input data and the
        reconstructed output data.

        Returns
        -------
            Keras tensor representing the loss.
        """

        inputs = self.model.get_input_at(0)
        outputs = self.model.get_output_at(-1)
        reconstr_loss = losses.mse(K.flatten(inputs), K.flatten(outputs))
        reconstr_loss *= self._input_shape[0] * self._input_shape[0]

        z_mean = self._encoder.get_output_at(-1)[0]
        z_log_var = self._encoder.get_output_at(-1)[1]

        kl_loss = 1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.1

        return K.mean(reconstr_loss + kl_loss)

    def losses(self):
        return {
            'decoder': self.loss_function,
        }

    def losses_weights(self):
        return {
            'decoder': 1
        }

    def predict(self, data):
        # Fixme
        # Override the base class implementation, which relies on the dataset
        # entity's get_test_set() method, not yet implemented for the
        # tractography dataset
        return self.model.predict(data)

    def predict_and_save(self, hdf5_file):

        # ToDo
        # Think what data should be saved
        # - Original DWI?
        # - Ground truth
        # - Prior if conditioning on some bundle

        name = 'none'
        dwi = None
        ground_truth = None
        prior = None

        # Use predicted mean instead of sampling (to avoid sampling outliers)
        latent_distribution = self._encoder.predict_on_batch(ground_truth)
        prediction = self._decoder.predict_on_batch(latent_distribution[:, :, 0])
        group = hdf5_file.create_group(name)
        # ToDo
        # When it will come the time to dealing with multiple subjects,
        # think of whether the dictionary should be created for all
        # subject, then call the method once.
        self.save_prediction(group, dwi, ground_truth, prediction, prior,
                             latent_distribution)