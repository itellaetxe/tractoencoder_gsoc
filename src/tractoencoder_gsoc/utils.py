# Adapted from the original code from the tractolearn repository: git@github.com:scil-vital/tractolearn.git
import sys
import argparse

import tensorflow as tf
import os
import numpy as np
import nibabel as nib
import h5py

from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import Streamlines  # same as nibabel.streamlines.ArraySequence

from keras import Layer, Sequential


dict_kernel_size_flatten_encoder_shape = {1: 12288,
                                          2: 10240,
                                          3: 8192,
                                          4: 7168,
                                          5: 5120}


class ReflectionPadding1D(Layer):
    def __init__(self, padding: int = 1, **kwargs):
        super(ReflectionPadding1D, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.padding, self.padding], [0, 0]],
                      mode='REFLECT')


def pre_pad(layer: Layer):
    return Sequential([ReflectionPadding1D(padding=1), layer])


def read_data(tractogram_fname: str, img_fname: str = None):
    # Load the anatomical data
    if img_fname is not None:
        img = nib.load(img_fname)
        img_header = img.header
    else:
        img_header = nib.Nifti1Header()

    # Load tractography data (assumes everything is resampled to 256 points)
    # from a TRK file
    to_space = Space.RASMM
    trk_header_check = True
    bbox_valid_check = True
    tractogram = load_tractogram(
        tractogram_fname,
        img_header,
        to_space=to_space,
        trk_header_check=trk_header_check,
        bbox_valid_check=bbox_valid_check,
    )
    strml = tractogram.streamlines

    return strml


def prepare_tensor_from_file(tractogram_fname: str, img_fname: str) -> tf.Tensor:
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
    else:
        img_header = nib.streamlines.trk.TrkFile.create_empty_header()

    # Save tractography data
    tractogram = nib.streamlines.Tractogram(streamlines=streamlines,
                                            affine_to_rasmm=np.eye(4))
    trkfile = nib.streamlines.TrkFile(tractogram, img_header)
    nib.streamlines.save(tractogram=trkfile,
                         filename=tractogram_fname)

    return None


def write_model_specs(spec_file: str, model, arguments) -> None:
    if not os.path.exists(os.path.dirname(spec_file)):
        os.makedirs(os.path.dirname(spec_file))

    with open(spec_file, "w") as f:
        f.write(f"### Model: {model.model.name}\n\n")
        f.write(f"### Input Data: {os.path.abspath(arguments.input_trk)}\n")
        f.write(f"### Anatomical Data: {os.path.abspath(arguments.input_anat)}\n")
        f.write(f"### Output Directory: {os.path.abspath(arguments.output_dir)}\n\n")
        f.write("### Training parameters:\n")
        f.write(f"## Batch Size: {arguments.batch_size}\n")
        f.write(f"## Epochs: {arguments.epochs}\n")
        f.write(f"## Learning Rate: {arguments.learning_rate}\n\n")
        f.write("### Model Parameters:\n")
        f.write(f"## Latent Space Dimensions: {model.latent_space_dims}\n")
        f.write(f"## Kernel Size: {model.kernel_size}\n\n")
        f.write("### Model Architecture:\n")
        for weight in model.model.weights:
            f.write(f"## Layer: {weight.path}\n")

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


# TODO: Translate function to TF
# def test_ae_model(tractogram_fname: str, img_fname: str, device):
#     # tractogram_fname is your trk, img_fname is the corresponding nii.gz anatomical reference file (e.g. T1w)
#     strml_tensor = prepare_tensor_from_file(tractogram_fname, img_fname)
#     strml_tensor = strml_tensor.to(device)

#     # Instantiate the model
#     model = AEModel()
#     # A full forward pass
#     y = model(strml_tensor)

#     print(f"Input shape: {strml_tensor.shape}")
#     print(f"Model: {model}")
#     print(f"Output shape: {y.shape}")

# TODO: Translate function to TF
# def test_ae_model_loader(tractogram_fname, img_fname, device):
#     # Use a loader for large datasets with which doing a complete forward and
#     # backward pass would not be possible due to GPU memory constraints.
#     # Note that in this case, all data is sent to the GPU, which may not be
#     # possible for large tractography files. That's why we used to use HDF5
#     # files and custom data loaders that would only fetch the batch size data
#     # from them.

#     # tractogram_fname is your trk, img_fname is the corresponding nii.gz anatomical reference file (e.g. T1w)
#     strml_tensor = prepare_tensor_from_file(tractogram_fname, img_fname)
#     strml_tensor = strml_tensor.to(device)

#     # Build a data loader
#     batch_size = 1000  # your batch_size
#     loader = torch.utils.data.DataLoader(
#         strml_tensor, batch_size=batch_size, shuffle=False
#     )

#     # Instantiate the model
#     model = AEModel()

#     reconstruction_loss = nn.MSELoss(reduction="sum")
#     loss = 0

#     strml_data = []
#     for _, data in enumerate(loader):

#         # Forward pass and compute loss
#         y = model(data)
#         batch_loss = reconstruction_loss(y, data)
#         loss += batch_loss.item()

#         # Undo the permutation
#         y = y.permute(0, 2, 1).detach().cpu().numpy()
#         strml_data.extend(y)

#     # Normalize the loss value to the size of the dataset
#     loss /= len(loader.dataset)

#     strml_rec = Streamlines(strml_data)

#     print(f"Input shape: {strml_tensor.shape}")
#     print(f"Model: {model}")
#     print(
#         f"Output shape: {strml_rec.shape}"
#     )  # Should be the same as input. Adapt, not sure if Streamlines has shape property

#     # Eventually, return strml_rec, loss in your scripts to check the loss and
#     # write the reconstructed streamlines if needed for visual inspection


# TODO: Translate function to TF
# def load_model_weights(weights_fname, device, lr, weight_decay):
#     # Instantiate the model
#     model = AEModel()

#     # Create the optimizer
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=lr,
#         weight_decay=weight_decay,
#     )

#     checkpoint = torch.load(weights_fname, map_location=device)
#     model.load_state_dict(checkpoint["state_dict"])

#     optimizer.load_state_dict(checkpoint["optimizer"])
#     epoch = model.load_state_dict(checkpoint["epoch"])  # best epoch
#     lowest_loss = checkpoint["lowest_loss"]

#     return model, optimizer, epoch, lowest_loss


# # TODO: Translate function to TF
# def train_ae_model(train_tractogram_fname, valid_tractogram_fname, img_fname, device, lr, weight_decay, epochs, weights_fname):
#     # Use a loader for large datasets that cannot be entirely hold in the GPU
#     # memory

#     # tractogram_fname is your trk, img_fname is the corresponding nii.gz anatomical reference file (e.g. T1w)
#     strml_tensor_train = prepare_tensor_from_file(train_tractogram_fname, img_fname)
#     strml_tensor_valid = prepare_tensor_from_file(valid_tractogram_fname, img_fname)

#     strml_tensor_train = strml_tensor_train.to(device)
#     strml_tensor_valid = strml_tensor_valid.to(device)

#     # Build the data loaders
#     batch_size = 1000  # your batch_size
#     train_loader = torch.utils.data.DataLoader(
#         strml_tensor_train, batch_size=batch_size, shuffle=False
#     )
#     valid_loader = torch.utils.data.DataLoader(
#         strml_tensor_valid, batch_size=batch_size, shuffle=False
#     )

#     # Instantiate the model
#     model = AEModel()

#     # Create the optimizer
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=lr,
#         weight_decay=weight_decay,
#     )

#     # Define the loss function
#     reconstruction_loss = nn.MSELoss(reduction="sum")

#     # As our loss is an MSE, worst possible loss is +infinity (or a very large number)
#     lowest_loss = sys.float_info.max

#     # Train model
#     for epoch in range(epochs):
#         # model.train()  # not sure if entirely necessary. We did not used to use it at the time.
#         train_loss = 0
#         valid_loss = 0
#         # Train split
#         for _, train_data in enumerate(train_loader):

#             optimizer.zero_grad()
#             train_y = model(train_data)
#             batch_loss = reconstruction_loss(train_y, train_data)

#             # Compute gradients and optimize model
#             batch_loss.backward()
#             train_loss += batch_loss.item()
#             optimizer.step()

#         # Normalize the loss value to the size of the dataset
#         train_loss /= len(train_loader.dataset)

#         # Validate
#         model.eval()
#         with torch.no_grad():
#             for _, valid_data in enumerate(valid_loader):

#                 valid_y = model(valid_data)
#                 batch_loss = reconstruction_loss(valid_y, valid_data)
#                 valid_loss += batch_loss.item()

#             # Normalize the loss value to the size of the dataset
#             valid_loss /= len(valid_loader.dataset)

#             # Save model and weights if loss is lower. Note that in our case the
#             # metric used to determine if a model is better is the
#             # reconstruction loss, but in other tasks we may choose a different
#             # metric, and the criterion might be "higher is better" (e.g. in a
#             # segmentation task, where the loss has been defined Dice+CE, we
#             # could define the validation split Dice as the metric).
#             if valid_loss < lowest_loss:
#                 lowest_loss = valid_loss
#                 # Note that we are storing the best epoch here, but not the last
#                 # epoch: thus, if the training is stopped, it will resume from
#                 # the best epoch. It should resume from the last epoch. But
#                 # at the time, we did not save this information. It should be
#                 # easy to store an additional key, value pair, as this is simply
#                 # a dictionary.
#                 state = {
#                     "epoch": epoch,
#                     "state_dict": model.state_dict(),
#                     "lowest_loss": lowest_loss,
#                     "optimizer": optimizer.state_dict(),
#                 }
#                 torch.save(state, weights_fname)