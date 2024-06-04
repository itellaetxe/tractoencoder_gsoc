# Adapted from the original code from the tractolearn repository: git@github.com:scil-vital/tractolearn.git

import tensorflow as tf
import os
import numpy as np
import nibabel as nib

import sys

from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import Streamlines  # same as nibabel.streamlines.ArraySequence


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

    strml_tensor = tf.Tensor(strml_data)

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