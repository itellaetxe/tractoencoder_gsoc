import os
import copy
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CallbackList

from tractoencoder_gsoc.data_loader import DataLoader
from tractoencoder_gsoc.prior import PriorFactory
from tractoencoder_gsoc.models import adv_ae_model


# Training step
# @tf.function
def train_step(model, data, y_labels, optimizers_dict,
               label_sample, real_dist, n_classes):
    # Unpack data
    x_batch = data

    # AE Gradient Tape
    with tf.GradientTape() as ae_tape:
        # Run the inputs through the AE (Encode->Decode)
        reconstruction = model.decoder(model.encoder(x_batch),
                                       training=True)
        # Compute Reconstruction Loss
        reconstruction_loss = tf.reduce_mean(tf.math.squared_difference(x_batch, reconstruction))
    # Compute the gradients of the AE
    ae_trainable_variables = model.encoder.trainable_variables + model.decoder.trainable_variables
    ae_grads = ae_tape.gradient(reconstruction_loss, ae_trainable_variables)
    optimizers_dict['ae'].apply_gradients(zip(ae_grads, ae_trainable_variables))

    # Discriminator
    with tf.GradientTape() as d_tape:
        # One hot encode the sampled labels
        label_sample_one_hot = tf.one_hot(label_sample, n_classes)
        real_dist_label = tf.concat([real_dist, label_sample_one_hot], axis=1)

        # Run the training data through the encoder to get the 'fake' latent vectors
        fake_dist = model.encoder(x_batch, training=True)
        # One hot encode the true labels
        label_fake_one_hot = tf.one_hot(y_labels.numpy(), n_classes)
        fake_dist_label = tf.concat([fake_dist, label_fake_one_hot], axis=1)

        # Run the latent vectors through the discriminator, 'fake' and 'real'
        _, real_logits = model.discriminator(real_dist_label, training=True)
        _, fake_logits = model.discriminator(fake_dist_label, training=True)
        # Discriminator loss
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_logits), logits=real_logits)
        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_logits), logits=fake_logits)
        D_loss = tf.reduce_mean(loss_real + loss_fake)

    # Compute the gradients of discriminator
    d_grads = d_tape.gradient(D_loss, model.discriminator.trainable_variables)
    optimizers_dict['discriminator'].apply_gradients(zip(d_grads,
                                                         model.discriminator.trainable_variables))

    with tf.GradientTape() as g_tape:
        encoder_output = model.encoder(x_batch, training=True)
        one_hot_labels = tf.one_hot(y_labels.numpy(), n_classes)
        encoder_output_label = tf.concat([encoder_output, one_hot_labels], axis=1)
        _, disc_fake_logits = model.discriminator(encoder_output_label, training=True)
        # Generator Loss
        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(disc_fake_logits), logits=disc_fake_logits))
    # Compute the gradients of generator
    g_grads = g_tape.gradient(G_loss, model.encoder.trainable_variables)
    optimizers_dict['encoder'].apply_gradients(zip(g_grads, model.encoder.trainable_variables))

    total_loss = reconstruction_loss + G_loss + D_loss

    return reconstruction_loss, D_loss, G_loss


# All training steps
def train_all_steps(
    model,
    optimizers_dict,
    train_ds,
    n_epochs,
    prior_type,
    n_classes,
    data_loader,
    prior_factory,
    output_dir
):
    """"Training of all batches `n_epochs` times.
    Creates and saves training results and logs
    Tensorboard metrics in the `log_dir` directory.
    """
    log_dir = os.path.join(output_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)
    _callbacks = [ModelCheckpoint(filepath=os.path.join(output_dir, "model_best.keras"),
                                  monitor='total_loss',
                                  save_weights_only=False,
                                  save_best_only=True,
                                  mode='auto',
                                  save_freq='epoch',
                                  verbose=1)]
    callbacks = CallbackList(_callbacks, add_history=True, model=model)
    logs = {}

    reconst_loss_vec = tf.metrics.Mean()
    discriminator_loss_vec = tf.metrics.Mean()
    encoder_loss_vec = tf.metrics.Mean()
    total_loss_vec = tf.metrics.Mean()
    latent_space_dims = model.encoder.output_shape[-1]

    # Manual best model save
    best_loss = np.inf
    prev_loss = np.inf
    best_model = {}  # To save the model and in which epoch it was saved
    best_model["model"] = None
    best_model["epoch"] = None

    # Training begins
    callbacks.on_train_begin(logs=logs)

    for epoch in range(n_epochs):
        # Epoch begins
        for batch_no, data in enumerate(train_ds):
            # Train step (batch) begins
            callbacks.on_batch_begin(batch=batch_no, logs=logs)
            callbacks.on_train_batch_begin(batch=batch_no, logs=logs)

            x_batch = data["streamlines"]
            y_labels = data["label"]
            attribute = data[data_loader.attribute_key]
            label_sample = np.random.randint(0, n_classes, size=[x_batch.shape[0]])
            real_distribution = prior_factory.get_prior(prior_type)(
                x_batch.shape[0], label_sample, n_classes, latent_space_dims
            )

            reconst_loss, D_loss, G_loss = train_step(
                model,
                x_batch,
                y_labels,
                optimizers_dict,
                label_sample,
                real_distribution,
                n_classes,
            )

            reconst_loss_vec(reconst_loss)
            discriminator_loss_vec(D_loss)
            encoder_loss_vec(G_loss)
            total_loss = reconst_loss + D_loss + G_loss
            total_loss_vec(total_loss)

            with summary_writer.as_default():
                tf.summary.scalar(
                    "ae_loss",
                    reconst_loss_vec.result(),
                    step=optimizers_dict["ae"].iterations,
                )
                tf.summary.scalar(
                    "encoder_loss",
                    encoder_loss_vec.result(),
                    step=optimizers_dict["encoder"].iterations,
                )
                tf.summary.scalar(
                    "discriminator_loss",
                    discriminator_loss_vec.result(),
                    step=optimizers_dict["discriminator"].iterations,
                )

            # Train step (batch) ends
            callbacks.on_batch_end(batch=batch_no, logs=logs)
            callbacks.on_train_batch_end(batch=batch_no, logs=logs)

        # Epoch ends
        callbacks.on_epoch_end(epoch=epoch, logs=logs)
        print(
            "Epoch: {} total_loss: {} gan_loss: {}, discriminator_loss: {} encoder_loss: {}".format(
                epoch,
                total_loss_vec.result(),
                reconst_loss_vec.result(),
                discriminator_loss_vec.result(),
                encoder_loss_vec.result(),
            )
        )
        # If the current loss is less than the best loss, save the model
        if total_loss_vec.result() < best_loss:
            best_loss = total_loss_vec.result()
            best_model["model"] = copy.deepcopy(model)
            best_model["epoch"] = epoch

    # Train ends
    callbacks.on_train_end(logs=logs)

    # When training ends, inform the user about the best model
    print(f"INFO: Best model happened at epoch {best_model['epoch']}")

    return best_model["model"]


def train_model(args):
    # Data
    data_loader = DataLoader(args.input_trk[0], args.batch_size)
    train_ds = data_loader.make_dataset()

    # Model
    adv_vae = adv_ae_model.JH_Adv_AE(latent_space_dims=args.latent_space_dims,
                                     kernel_size=args.kernel_size,
                                     n_classes=data_loader.n_classes)
    prior_factory = PriorFactory(data_loader.n_classes)
    # Optimizers
    optimizers_dict = {
        "encoder": tf.optimizers.Adam(learning_rate=args.learning_rate),
        "discriminator": tf.optimizers.Adam(learning_rate=args.learning_rate / 5),
        "ae": tf.optimizers.Adam(learning_rate=args.learning_rate),
    }

    # Training
    best_model = train_all_steps(
        adv_vae,
        optimizers_dict,
        train_ds,
        args.epochs,
        args.prior_type,
        data_loader.n_classes,
        data_loader,
        prior_factory,
        args.output_dir,
    )

    return best_model
