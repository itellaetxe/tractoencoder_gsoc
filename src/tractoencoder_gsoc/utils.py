# Adapted from the original code from the tractolearn repository: git@github.com:scil-vital/tractolearn.git
import sys
import argparse

import tensorflow as tf
import os
import numpy as np
import nibabel as nib
import h5py

from dipy.io.stateful_tractogram import Space
from dipy.io.utils import (get_reference_info,
                           is_header_compatible,
                           create_nifti_header)
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import Streamlines  # same as nibabel.streamlines.ArraySequence

from keras import Layer, Sequential

dict_kernel_size_flatten_encoder_shape = {1: 12288,
                                          2: 10240,
                                          3: 8192,
                                          4: 7168,
                                          5: 5120}


def safe_exp(x):
    # Safe exp operation to prevent exp from producing inf values
    return tf.clip_by_value(tf.exp(x), -1e10, 1e10)


def cross_entropy():
    return tf.nn.sigmoid_cross_entropy_with_logits(from_logits=True)


class ReflectionPadding1D(Layer):
    def __init__(self, padding: int = 1, **kwargs):
        super(ReflectionPadding1D, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.padding, self.padding], [0, 0]],
                      mode='REFLECT')


def pre_pad(layer: Layer):
    return Sequential([ReflectionPadding1D(padding=1), layer])


def read_data(tractogram_fname: str, img_fname: str = None,
              trk_header_check: bool = False,
              bbox_valid_check: bool = False):
    # Load the anatomical data
    if img_fname is None:
        img_header = nib.Nifti1Header()
    else:
        img = nib.load(img_fname)
        img_header = img.header

    # Load tractography data (assumes everything is resampled to 256 points)
    # from a TRK file
    to_space = Space.RASMM
    tractogram = load_tractogram(
        tractogram_fname,
        img_header,
        to_space=to_space,
        trk_header_check=trk_header_check,
        bbox_valid_check=bbox_valid_check,
    )
    strml = tractogram.streamlines

    return strml


def compute_streamline_length(streamline):
    # Calculate differences between consecutive points
    diffs = np.diff(streamline, axis=0)

    # Compute the Euclidean distance between consecutive points
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

    # Sum the distances to get the total length
    total_length = np.sum(distances)

    return total_length


def prepare_tensor_from_file(tractogram_fname: str,
                             img_fname: str) -> tf.Tensor:
    strml = read_data(tractogram_fname, img_fname)

    # Dump streamline data to array
    strml_data = np.vstack([strml[i][np.newaxis,] for i in range(len(strml))])

    # Build a tf.Tensor using the streamline data: should be an array that
    # has N rows, each row having 256 columns

    strml_tensor = tf.convert_to_tensor(strml_data)

    return strml_tensor


def save_tractogram(streamlines: np.array,
                    tractogram_fname: str,
                    img_fname: str = None) -> None:

    # Load the anatomical data
    if img_fname is not None:
        img = nib.load(img_fname)
        img_header = img.header
        affine = img_header.get_base_affine()
    else:
        img_header = nib.streamlines.trk.TrkFile.create_empty_header()

    # Save tractography data
    tractogram = nib.streamlines.Tractogram(streamlines=streamlines,
                                            affine_to_rasmm=affine)
    trk_file = nib.streamlines.TrkFile(tractogram=tractogram,
                                       header=img_header)
    nib.streamlines.save(tractogram=trk_file,
                         filename=tractogram_fname)

    return None


def write_model_specs(spec_file: str, model, arguments,
                      train_history=None) -> None:
    if not os.path.exists(os.path.dirname(spec_file)):
        os.makedirs(os.path.dirname(spec_file))

    with open(spec_file, "w") as f:
        if hasattr(model, "model"):
            f.write(f"### Model: {model.model.name}\n\n")
        else:
            f.write(f"### Model: {model.name}\n\n")

        if len(arguments.input_trk) > 1:
            f.write("### Input Data:\n")
            for trk_path in arguments.input_trk:
                f.write(f"## {os.path.abspath(trk_path)}\n")
        else:
            f.write(f"### Input Data: {os.path.abspath(arguments.input_trk[0])}\n")

        f.write(f"### Anatomical Data: {os.path.abspath(arguments.input_anat)}\n")
        f.write(f"### Output Directory: {os.path.abspath(arguments.output_dir)}\n\n")
        f.write("### Training parameters:\n")
        f.write(f"## Batch Size: {arguments.batch_size}\n")
        f.write(f"## Epochs: {arguments.epochs}\n")
        f.write(f"## Learning Rate: {arguments.learning_rate}\n\n")
        if hasattr(arguments, "beta"):
            f.write(f"## Beta weight of KL loss: {arguments.beta}\n\n")
        f.write("### Model Parameters:\n")
        f.write(f"## Latent Space Dimensions: {model.latent_space_dims}\n")
        f.write(f"## Kernel Size: {model.kernel_size}\n\n")
        f.write("### Model Architecture:\n")

        if hasattr(model, "model"):
            for weight in model.model.weights:
                f.write(f"## Layer: {weight.path}\n")
        else:
            for weight in model.weights:
                f.write(f"## Layer: {weight.path}\n")

    # Write the training history to the same file
    history_keys = list(train_history.history.keys())
    hist = train_history.history
    epochs = train_history.epoch
    history_text = ""
    for i in range(len(epochs)):
        history_text += f"[Epoch {epochs[i]}] "
        for key in history_keys:
            history_text += (f"{key} = {str(hist[key][i])[:11]} || ")
        history_text += "\n"

    if train_history is not None:
        with open(spec_file, "a") as f:
            f.write("\n### Training History:\n")
            f.write(history_text + "\n\n")
    print(f"INFO: Model specs written to {spec_file}")

    return None


def load_h5_dataset(h5_fname: str, dataset_name: str = None) -> np.array:
    with h5py.File(h5_fname, "r") as f:
        data = f[dataset_name][()]

    return data


def process_arguments_hdf5() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the autoencoder model using HDF5 datasets.")
    parser.add_argument("--input_dataset", type=str, help="Path to the input HDF5 file (.h5 file)")
    parser.add_argument("--input_anat", type=str, help="Path to the input anatomical image (NIfTI file)")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory where results will be saved")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training the model")
    parser.add_argument("--latent_space_dims", type=int, default=32, help="Number of dimensions in the latent space")
    parser.add_argument("--kernel_size", type=int, default=3, help="Size of the kernel for the convolutional layers")
    parser.add_argument("--learning_rate", type=float, default=0.00068, help="Learning rate for the optimizer")
    parser.add_argument("--seed", type=int, default=2208, help="Seed for reproducibility")
    args = parser.parse_args()

    # Sanity check of CLI arguments
    if not os.path.exists(args.input_dataset):
        raise FileNotFoundError(f"Input dataset not found at {args.input_dataset}")
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


def process_arguments_trk() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the autoencoder model")
    parser.add_argument("--input_trk", nargs='+', type=str, help="Path(s) to the input tractogram(s) (.trk)")
    parser.add_argument("--input_anat", type=str, help="Path to the input anatomical image (NIfTI)")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory where results will be saved")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training the model")
    parser.add_argument("--latent_space_dims", type=int, default=32, help="Number of dimensions in the latent space")
    parser.add_argument("--kernel_size", type=int, default=3, help="Size of the kernel for the convolutional layers")
    parser.add_argument("--learning_rate", type=float, default=0.00068, help="Learning rate for the optimizer")
    parser.add_argument("--seed", type=int, default=2208, help="Seed for reproducibility")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for the KL divergence loss")
    args = parser.parse_args()

    # Sanity check of CLI arguments
    for trk_path in args.input_trk:
        if not os.path.exists(trk_path):
            raise FileNotFoundError(f"Input dataset not found at {trk_path}")
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


class UpdateEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(UpdateEpochCallback, self).__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        # Update the model's current_epoch property at the start of each epoch
        self.model.current_epoch.assign(epoch)
