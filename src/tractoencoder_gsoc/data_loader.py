# Adapted from: https://github.com/elsanns/adversarial-autoencoder-tf2/blob/master/utils/data_loader.py
"""Loading data from tensorflow_datasets, creating train datasets."""


import pickle

import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, path_to_pkl, batch_size):
        super(DataLoader, self).__init__()

        self.pkl_data = pickle.load(open(path_to_pkl, "rb"))
        self.batch_size = batch_size
        self.streamline_data = self.pkl_data["streamlines"]
        self.label_data = self.pkl_data["label"]
        keys = list(self.pkl_data.keys())
        keys.remove("streamlines")
        keys.remove("label")

        self.attribute_key = keys[0]
        self.attribute_data = self.pkl_data[self.attribute_key]

        # Number of bundles
        self.n_classes = len(np.unique(self.pkl_data["label"]))

        self.train_ds = None

    def make_dataset(self):
        """Constructs training dataset.

        Returns:
            tf.data.Dataset:
                Training dataset
        """

        # Create a tf.data.Dataset from the dictionary
        train_ds = tf.data.Dataset.from_tensor_slices(self.pkl_data)

        # Check that it was converted correctly, shuffle, and group into batches
        assert isinstance(train_ds, tf.data.Dataset)
        train_ds.shuffle(50000)

        train_ds = train_ds.batch(self.batch_size)

        self.train_ds = train_ds
        return self.train_ds
