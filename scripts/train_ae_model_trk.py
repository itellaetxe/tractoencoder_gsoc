import argparse
import os
import numpy as np
import nibabel as nib

from tractoencoder_gsoc import utils as utils
from tractoencoder_gsoc import ae_model


def process_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the autoencoder model")
    parser.add_argument("--input_trk", type=str, help="Path to the input tractogram (.trk file)")
    parser.add_argument("--input_anat", type=str, help="Path to the input anatomical image (NIfTI file)")
    parser.add_argument("--output_model", type=str, help="Path to the output model")
    args = parser.parse_args()
    if not os.path.exists(args.input_trk):
        raise FileNotFoundError(f"Input dataset not found at {args.input_dataset}")
    elif not os.path.exists(args.input_anat):
        raise FileNotFoundError(f"Input anatomical image not found at {args.input_anat}")
    elif not os.path.exists(args.output_model):
        print(f"WARNING: Output model not found at {args.output_model}")
        try:
            os.mkdir(args.output_model)
        except (FileNotFoundError, FileExistsError) as e:
            if e.__class__ == FileNotFoundError:
                print(f"WARNING: Output model directory not found at {args.output_model}")
                print("INFO: Creating the output directory")
                os.makedirs(args.output_model)
            elif e.__class__ == FileExistsError:
                print(f"WARNING: Output model directory not found at {args.output_model}")
    elif os.path.isdir(args.output_model):
        print(f"The output_model is not a directory. Your model will be stored at {os.path.join(args.output_model, 'model.weights.h5')}")
        args.output_model = os.path.join(args.output_model, "model.weights.h5")

    return args


if __name__ == "__main__":
    # Example of using the model
    latent_space_dims = 32
    kernel_size = 3
    model = ae_model.IncrFeatStridedConvFCUpsampReflectPadAE(latent_space_dims=latent_space_dims,
                                                             kernel_size=kernel_size)

    # Get input arguments
    args = process_arguments()

    # Read the data
    input_data = np.array(utils.read_data(args.input_trk, args.input_anat))

    # Train the model
    model.fit(x=input_data, y=input_data, epochs=10, batch_size=100)

    # Save the model
    model.save_weights(args.output_model)

    # Run the input data through the model, convert it to a np.array
    y = model(input_data).numpy()

    # Save the tractogram
    trk_fname = os.path.join(os.path.dirname(args.output_model), "output.trk")
    utils.save_tractogram(streamlines=y,
                          img_fname=args.input_anat,
                          tractogram_fname=trk_fname)
