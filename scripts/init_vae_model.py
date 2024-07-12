import os
import subprocess

import numpy as np
import tensorflow as tf

import tractoencoder_gsoc.utils as utils
from tractoencoder_gsoc.models import vae_model

if __name__ == '__main__':
    tf.random.set_seed(2208)
    # Example of using the model
    latent_space_dims = 32
    kernel_size = 3
    model = vae_model.IncrFeatStridedConvFCUpsampReflectPadVAE(latent_space_dims=latent_space_dims,
                                                               kernel_size=kernel_size)

    # Read data
    wd = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(wd, "data/fibercup/")
    streamline_path = os.path.join(data_path, "fibercup_advanced_filtering_no_ushapes/ae_input_std_endpoints/test/fibercup_Simulated_prob_tracking_minL10_resampled256_plausibles_std_endpoints_test.trk")
    anat_path = os.path.join(data_path, "Simulated_FiberCup.nii.gz")
    input_data = utils.prepare_tensor_from_file(streamline_path, anat_path).numpy()

    # Example of using the model
    output = model(input_data)
    model.summary()

    # Define the TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    # Try the train method (x and y should be the same)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000668))
    model.fit(x=input_data, epochs=100, batch_size=300,
              callbacks=[tensorboard_callback])

    # Compute reconstruction after training, and plot it
    reconstruction = model(input_data).numpy()
    # Write to tractogram
    output_trk_fname = os.path.join(wd, "output.trk")
    utils.save_tractogram(streamlines=reconstruction,
                          img_fname=anat_path,
                          tractogram_fname=output_trk_fname)
