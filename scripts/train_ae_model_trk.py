import argparse
import os
import numpy as np
import nibabel as nib
import subprocess

from tractoencoder_gsoc import utils as utils
from tractoencoder_gsoc import ae_model


def process_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the autoencoder model")
    parser.add_argument("--input_trk", type=str, help="Path to the input tractogram (.trk file)")
    parser.add_argument("--input_anat", type=str, help="Path to the input anatomical image (NIfTI file)")
    parser.add_argument("--output_model", type=str, help="Path to the output model")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training the model")
    parser.add_argument("--latent_space_dims", type=int, default=32, help="Number of dimensions in the latent space")
    parser.add_argument("--kernel_size", type=int, default=3, help="Size of the kernel for the convolutional layers")
    args = parser.parse_args()

    # Sanity check of CLI arguments
    if not os.path.exists(args.input_trk):
        raise FileNotFoundError(f"Input dataset not found at {args.input_trk}")
    elif not os.path.exists(args.input_anat):
        raise FileNotFoundError(f"Input anatomical image not found at {args.input_anat}")
    elif not os.path.exists(args.output_model):
        print(f"WARNING: Output model not found at {args.output_model}")
        try:
            os.makedirs(args.output_model)
        except (FileNotFoundError, FileExistsError) as e:
            if e.__class__ == FileNotFoundError:
                print(f"WARNING: Output model directory not found at {args.output_model}")
                print("INFO: Creating the output directory")
                os.makedirs(args.output_model)
            elif e.__class__ == FileExistsError:
                print(f"WARNING: Output model directory found at {args.output_model}")
        finally:
            if os.path.isdir(args.output_model):
                print(f"INFO: The output_model is a directory. Your model will be stored at {os.path.join(args.output_model, 'model.weights.h5')}")
                args.output_model = os.path.join(args.output_model, "model.weights.h5")

    return args


def write_model_specs(spec_file: str, model) -> None:
    if not os.path.exists(os.path.dirname(spec_file)):
        os.makedirs(os.path.dirname(spec_file))

    with open(spec_file, "w") as f:
        f.write(f"### Model: {model.model.name}\n\n")
        f.write("### Training parameters:\n")
        f.write(f"## Batch Size: {args.batch_size}\n")
        f.write(f"## Epochs: {args.epochs}\n\n")
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

    # Example of using the model
    model = ae_model.IncrFeatStridedConvFCUpsampReflectPadAE(latent_space_dims=args.latent_space_dims,
                                                             kernel_size=args.kernel_size)
    # Read the data
    input_data = np.array(utils.read_data(args.input_trk, args.input_anat))

    # Train the model
    model.fit(x=input_data, y=input_data, epochs=args.epochs, batch_size=args.batch_size)

    # Save the model
    model.save_weights(args.output_model)

    # Run the input data through the model, convert it to a np.array
    y = model(input_data).numpy()

    # Save the tractogram
    output_trk_fname = os.path.join(os.path.dirname(args.output_model), "output.trk")
    print(f"INFO: Saving the reconstructed tractogram at {output_trk_fname}")
    utils.save_tractogram(streamlines=y,
                          img_fname=args.input_anat,
                          tractogram_fname=output_trk_fname)

    # Write the model specs to a file for future reference
    spec_file = os.path.join(os.path.dirname(args.output_model), "model_specs.txt")

    write_model_specs(spec_file=spec_file, model=model)

    # Save a screenshot of the input and the output using dipy_horizon
    input_screenshot_fname = os.path.join(os.path.dirname(args.output_model), "input_view.png")
    output_screenshot_fname = os.path.join(os.path.dirname(args.output_model), "output_view.png")
    print(f"INFO: Saving the input screenshot at {input_screenshot_fname}")

    # Just in case if fury is not installed
    subprocess.run(["uv pip install -U fury"], shell=True)
    command = f"dipy_horizon {args.input_trk} --stealth --out_stealth_png {input_screenshot_fname}"
    subprocess.run(command, shell=True)
    command = f"dipy_horizon {output_trk_fname} --stealth --out_stealth_png {output_screenshot_fname}"
    subprocess.run(command, shell=True)
