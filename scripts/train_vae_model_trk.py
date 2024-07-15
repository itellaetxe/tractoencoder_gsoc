import argparse
import os
import numpy as np
import nibabel as nib
import subprocess

import tensorflow as tf

from tractoencoder_gsoc import utils as utils
from tractoencoder_gsoc.models import vae_model


if __name__ == "__main__":

    # Get input arguments
    args = utils.process_arguments_trk()

    # Set the seed for reproducibility
    tf.random.set_seed(args.seed)

    # Example of using the model
    model = vae_model.IncrFeatStridedConvFCUpsampReflectPadVAE(latent_space_dims=args.latent_space_dims,
                                                             kernel_size=args.kernel_size)

    # Read data. If multiple trk files, read all and concatenate in numpy array along 1st axis
    if len(args.input_trk) == 1:
        input_data = utils.prepare_tensor_from_file(args.input_trk, args.input_anat)
    else:
        input_data = []
        for trk_path in args.input_trk:
            input_data.append(utils.prepare_tensor_from_file(trk_path, args.input_anat).numpy())
        # Concatenate along the first axis
        input_data = tf.convert_to_tensor(np.concatenate(input_data, axis=0))

    # Compile the model, then fit it (train it)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.MeanSquaredError())
    model.fit(x=input_data,
              epochs=args.epochs,
              batch_size=args.batch_size,
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir=args.output_dir)])

    # Save the model
    model_fname = os.path.join(args.output_dir, "model.weights.h5")
    model.save_weights(model_fname)
    model.save(os.path.join(args.output_dir, "model_final.keras"))

    # Run the input data through the model, convert it to a np.array
    y = model(input_data).numpy()

    # Save the tractogram
    output_trk_fname = os.path.join(args.output_dir, "output.trk")
    print(f"INFO: Saving the reconstructed tractogram at {output_trk_fname}")
    utils.save_tractogram(streamlines=y,
                          img_fname=args.input_anat,
                          tractogram_fname=output_trk_fname)

    # Write the model specs to a file for future reference
    spec_file = os.path.join(args.output_dir, "model_specs.txt")
    utils.write_model_specs(spec_file=spec_file, model=model, arguments=args)

    # Save a screenshot of the input and the output using dipy_horizon
    input_screenshot_fname = os.path.join(args.output_dir, "input_view.png")
    output_screenshot_fname = os.path.join(args.output_dir, "output_view.png")
    print(f"INFO: Saving the input screenshot at {input_screenshot_fname}")

    activate_env = "source " + os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv/bin/activate")
    current_env = os.environ.copy()
    # Just in case if fury is not installed
    # command = f"{activate_env} && uv pip install -U fury && uv pip install numpy==1.26.0"
    # subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
    command = f"{activate_env} && dipy_horizon {args.input_trk} --stealth --out_stealth_png {input_screenshot_fname}"
    subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
    command = f"{activate_env} && dipy_horizon {output_trk_fname} --stealth --out_stealth_png {output_screenshot_fname}"
    subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
