import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer


def mse(model, original):
    return tf.reduce_mean(tf.square(tf.subtract(model(original), original)))


def train_autoencoder(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(
            loss(model, original), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)


def log_results(model, X, max_outputs, epoch, prefix):
    loss_values = mse(model, X)

    sample_img = X[sample(range(X.shape[0]), max_outputs), :]
    original = tf.reshape(sample_img, (max_outputs, 28, 28, 1))
    encoded = tf.reshape(
        model.encode(sample_img), (sample_img.shape[0], 8, 8, 1))
    decoded = tf.reshape(
        model(tf.constant(sample_img)), (sample_img.shape[0], 28, 28, 1))
    tf.summary.scalar("{}_loss".format(prefix), loss_values, step=epoch + 1)
    tf.summary.image(
        "{}_original".format(prefix),
        original,
        max_outputs=max_outputs,
        step=epoch + 1)
    tf.summary.image(
        "{}_encoded".format(prefix),
        encoded,
        max_outputs=max_outputs,
        step=epoch + 1)
    tf.summary.image(
        "{}_decoded".format(prefix),
        decoded,
        max_outputs=max_outputs,
        step=epoch + 1)

    return loss_values


def preprocess_mnist(batch_size):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train / np.max(X_train)
    X_train = X_train.reshape(X_train.shape[0],
                              X_train.shape[1] * X_train.shape[2]).astype(
                                  np.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(
        batch_size)

    y_train = y_train.astype(np.int32)
    train_labels = tf.data.Dataset.from_tensor_slices(y_train).batch(
        batch_size)

    X_test = X_test / np.max(X_test)
    X_test = X_test.reshape(
        X_test.shape[0], X_test.shape[1] * X_test.shape[2]).astype(np.float32)

    y_test = y_test.astype(np.int32)

    return X_train, X_test, train_dataset, y_train, y_test, train_labels


class Encoder(Layer):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.output_layer = Dense(units=self.units, activation=tf.nn.relu)

    @tf.function
    def call(self, X):
        return self.output_layer(X)


class Decoder(Layer):
    def __init__(self, encoder):
        super(Decoder, self).__init__()
        self.encoder = encoder

    def build(self, input_shape):
        self.output_layer = Dense(units=self.encoder.input_shape)

    @tf.function
    def call(self, X):
        return self.output_layer(X)


class AutoEncoder(Model):
    def __init__(self, units):
        super(AutoEncoder, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.encoder = Encoder(units=self.units)
        self.encoder.build(input_shape)
        self.decoder = Decoder(encoder=self.encoder)

    @tf.function
    def call(self, X):
        Z = self.encoder(X)
        return self.decoder(Z)

    @tf.function
    def encode(self, X):
        return self.encoder(X)

    @tf.function
    def decode(self, Z):
        return self.decode(Z)


def test_autoencoder(batch_size,
                     learning_rate,
                     epochs,
                     max_outputs=4,
                     seed=None):

    tf.random.set_seed(seed)

    X_train, X_test, train_dataset, _, _, _ = preprocess_mnist(
        batch_size=batch_size)

    autoencoder = AutoEncoder(units=64)
    opt = tf.optimizers.Adam(learning_rate=learning_rate)

    log_path = 'logs/autoencoder'
    if os.path.exists(log_path):
        shutil.rmtree(log_path)

    writer = tf.summary.create_file_writer(log_path)

    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                for step, batch in enumerate(train_dataset):
                    train_autoencoder(mse, autoencoder, opt, batch)

                # logs (train)
                train_loss = log_results(
                    model=autoencoder,
                    X=X_train,
                    max_outputs=max_outputs,
                    epoch=epoch,
                    prefix='train')

                # logs (test)
                test_loss = log_results(
                    model=autoencoder,
                    X=X_test,
                    max_outputs=max_outputs,
                    epoch=epoch,
                    prefix='test')

                writer.flush()

                template = 'Epoch {}, Train loss: {:.5f}, Test loss: {:.5f}'
                print(
                    template.format(epoch + 1, train_loss.numpy(),
                                    test_loss.numpy()))

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    np.savez_compressed('saved_models/encoder.npz',
                        *autoencoder.encoder.get_weights())


if __name__ == '__main__':
    test_autoencoder(batch_size=128, learning_rate=1e-3, epochs=20, seed=42)