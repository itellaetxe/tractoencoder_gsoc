import argparse
import os
import numpy as np
import nibabel as nib
import subprocess

import tensorflow as tf

from tractoencoder_gsoc import utils as utils
from tractoencoder_gsoc.trainers import adv_vae_trainer
from tractoencoder_gsoc.models import adv_ae_model

# Code partially inspired from:
# https://github.com/elsanns/adversarial-autoencoder-tf2
# (Very nice and neat implementation)


def main(args):
    # Set the seed for reproducibility
    tf.random.set_seed(args.seed)

    adv_vae_trainer.train_model(args)


if __name__ == "__main__":
    # Get input arguments
    args = utils.process_arguments_trk()

    if args is None:
        exit()

    main(args)

    # # Save the training history
    # train_history = model.fit(x=[input_streamlines, streamline_lengths],
    #                           epochs=args.epochs,
    #                           batch_size=args.batch_size,
    #                           callbacks=[tensorboard_cb, early_stopping_monitor])

    # # Run the input data through the model, convert it to a np.array
    # y = model(input_streamlines).numpy()

    # # Save the model
    # model_fname = os.path.join(args.output_dir, "model.weights.h5")
    # model.save_weights(model_fname)
    # model.save(os.path.join(args.output_dir, "model_final.keras"))

    # # Save the tractogram
    # output_trk_fname = os.path.join(args.output_dir, "output.trk")
    # print(f"INFO: Saving the reconstructed tractogram at {output_trk_fname}")
    # utils.save_tractogram(streamlines=y,
    #                       img_fname=args.input_anat,
    #                       tractogram_fname=output_trk_fname)

    # # Write the model specs to a file for future reference
    # spec_file = os.path.join(args.output_dir, "model_specs.txt")
    # utils.write_model_specs(spec_file=spec_file, model=model, arguments=args,
    #                         train_history=train_history)

    # # Save a screenshot of the input and the output using dipy_horizon
    # input_screenshot_fname = os.path.join(args.output_dir, "input_view.png")
    # output_screenshot_fname = os.path.join(args.output_dir, "output_view.png")
    # print(f"INFO: Saving the input screenshot at {input_screenshot_fname}")

    # activate_env = "source " + os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv/bin/activate")
    # current_env = os.environ.copy()
    # # Just in case if fury is not installed
    # # command = f"{activate_env} && uv pip install -U fury && uv pip install numpy==1.26.0"
    # # subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
    # command = f"{activate_env} && dipy_horizon {args.input_trk} --stealth --out_stealth_png {input_screenshot_fname}"
    # subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
    # command = f"{activate_env} && dipy_horizon {output_trk_fname} --stealth --out_stealth_png {output_screenshot_fname}"
    # subprocess.run(command, env=current_env, executable="/bin/bash", shell=True)
