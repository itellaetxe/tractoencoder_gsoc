import argparse
import os
import numpy as np
import nibabel as nib
import subprocess

import tensorflow as tf

from tractoencoder_gsoc import utils as utils
from tractoencoder_gsoc.trainers import adv_vae_trainer
from tractoencoder_gsoc.models import adv_ae_model
from tractoencoder_gsoc.data_loader import DataLoader

# Code partially inspired from:
# https://github.com/elsanns/adversarial-autoencoder-tf2
# (Very nice and neat implementation)


def main(args):
    # Set the seed for reproducibility
    tf.random.set_seed(args.seed)

    best_model = adv_vae_trainer.train_model(args)

    # Inference with the model
    data_loader = DataLoader(args.input_trk[0], args.batch_size)
    input_streamlines = np.array(data_loader.streamline_data)

    # Run input_streamlines through the model
    y = best_model(input_streamlines).numpy()

    # Save the model weights and the model
    model_fname = os.path.join(args.output_dir, "model.weights.h5")
    best_model.save_weights(model_fname)
    best_model.save(os.path.join(args.output_dir, "model_final_save.keras"))

    # Save the tractogram
    output_trk_fname = os.path.join(args.output_dir, "output.trk")
    print(f"INFO: Saving the reconstructed tractogram at {output_trk_fname}")
    utils.save_tractogram(streamlines=y,
                          img_fname=args.input_anat,
                          tractogram_fname=output_trk_fname)

    # Write the model specs to a file for future reference
    spec_file = os.path.join(args.output_dir, "model_specs.txt")
    utils.write_model_specs(spec_file=spec_file, model=best_model,
                            arguments=args, train_history=None)


if __name__ == "__main__":
    # Get input arguments
    args = utils.process_arguments_trk()

    if args is None:
        exit()

    main(args)
