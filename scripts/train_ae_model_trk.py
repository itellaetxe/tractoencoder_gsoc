import argparse
import os
import numpy as np
import nibabel as nib
import subprocess

import tensorflow as tf

from tractoencoder_gsoc import utils as utils
from tractoencoder_gsoc import ae_model


def process_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the autoencoder model")
    parser.add_argument("--input_trk", type=str, help="Path to the input tractogram (.trk file)")
    parser.add_argument("--input_anat", type=str, help="Path to the input anatomical image (NIfTI file)")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory where results will be saved")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training the model")
    parser.add_argument("--latent_space_dims", type=int, default=32, help="Number of dimensions in the latent space")
    parser.add_argument("--kernel_size", type=int, default=3, help="Size of the kernel for the convolutional layers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--seed", type=int, default=2208, help="Seed for reproducibility")
    args = parser.parse_args()

    # Sanity check of CLI arguments
    if not os.path.exists(args.input_trk):
        raise FileNotFoundError(f"Input dataset not found at {args.input_trk}")
    if not os.path.exists(args.input_anat):
        raise FileNotFoundError(f"Input anatomical image not found at {args.input_anat}")

    if os.path.exists(args.output_dir):
        # If the output directory exists and it is NOT empty, raise Error because we do not want to overwrite
        if len(os.listdir(args.output_dir)) != 0:
            raise FileExistsError(f"Output directory {args.output_dir} is not empty. Please provide an empty directory")
        else:
            print(f"WARNING: Empty output directory found at {args.output_dir}")
    # If the output directory does not exist, create it:
    else:
        os.makedirs(args.output_dir)

    print(f"INFO: Your results will be stored at {args.output_dir}")

    return args


def write_model_specs(spec_file: str, model) -> None:
    if not os.path.exists(os.path.dirname(spec_file)):
        os.makedirs(os.path.dirname(spec_file))

    with open(spec_file, "w") as f:
        f.write(f"### Model: {model.model.name}\n\n")
        f.write("### Training parameters:\n")
        f.write(f"## Batch Size: {args.batch_size}\n")
        f.write(f"## Epochs: {args.epochs}\n")
        f.write(f"## Learning Rate: {args.learning_rate}\n\n")
        f.write("### Model Parameters:\n")
        f.write(f"## Latent Space Dimensions: {model.latent_space_dims}\n")
        f.write(f"## Kernel Size: {model.kernel_size}\n\n")
        f.write("### Model Architecture:\n")
        for weight in model.model.weights:
            f.write(f"## Layer: {weight.path}\n")

    print(f"INFO: Model specs written to {spec_file}")

    return None


if __name__ == "__main__":

    # Get input arguments
    args = process_arguments()

    # Set the seed for reproducibility
    tf.random.set_seed(args.seed)

    # Example of using the model
    model = ae_model.IncrFeatStridedConvFCUpsampReflectPadAE(latent_space_dims=args.latent_space_dims,
                                                             kernel_size=args.kernel_size)
    # Read the data
    input_data = utils.prepare_tensor_from_file(args.input_trk, args.input_anat)

    # Train the model (first compile it)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.MeanSquaredError())
    model.fit(x=input_data, y=input_data, epochs=args.epochs, batch_size=args.batch_size)

    # Save the model
    model_fname = os.path.join(args.output_dir, "model.weights.h5")
    model.save_weights(model_fname)

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

    write_model_specs(spec_file=spec_file, model=model)

    # Save a screenshot of the input and the output using dipy_horizon
    input_screenshot_fname = os.path.join(args.output_dir, "input_view.png")
    output_screenshot_fname = os.path.join(args.output_dir, "output_view.png")
    print(f"INFO: Saving the input screenshot at {input_screenshot_fname}")

    # Just in case if fury is not installed
    activate_env = "source " + os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv/bin/activate")
    current_env = os.environ.copy()
    command = f"{activate_env} && uv pip install -U fury"
    subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
    command = f"{activate_env} && dipy_horizon {args.input_trk} --stealth --out_stealth_png {input_screenshot_fname}"
    subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
    command = f"{activate_env} && dipy_horizon {output_trk_fname} --stealth --out_stealth_png {output_screenshot_fname}"
    subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
