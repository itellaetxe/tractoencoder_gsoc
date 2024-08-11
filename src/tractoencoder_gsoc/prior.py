# Taken from: https://github.com/elsanns/adversarial-autoencoder-tf2/blob/master/utils/prior_utils.py
"""Generates samples from supported types of prior distributions."""

import numpy as np
from math import sin, cos, sqrt


class PriorFactory:
    """Class containing methods for generation of samples
    from supported prior distributions.
    """

    def __init__(self, n_classes, gm_x_stddev=0.5, gm_y_stddev=0.1):
        super(PriorFactory, self).__init__()
        self.n_classes = n_classes
        self.gaussian_mixture_x_stddev = gm_x_stddev
        self.gaussian_mixture_y_stddev = gm_y_stddev

    def gaussian_mixture(self, batch_size, labels, n_classes, dims):
        x_stddev = self.gaussian_mixture_x_stddev
        y_stddev = self.gaussian_mixture_y_stddev
        shift = 3 * x_stddev

        # Generate the initial Gaussian distributed samples
        z = np.random.normal(0, [x_stddev] + [y_stddev] * (dims - 1), (batch_size, dims)).astype("float32")
        z[:, 0] += shift  # Shift only the first dimension

        def rotate(z, label):
            angle = label * 2.0 * np.pi / n_classes
            rotation_matrix = np.eye(dims)
            # Apply rotation in the first 2 dimensions only
            rotation_matrix[0, 0] = cos(angle)
            rotation_matrix[0, 1] = -sin(angle)
            rotation_matrix[1, 0] = sin(angle)
            rotation_matrix[1, 1] = cos(angle)

            z[np.where(labels == label)] = np.dot(z[np.where(labels == label)], rotation_matrix)
            return z

        # Apply rotation for each class label
        for label in set(labels):
            z = rotate(z, label)

        return z

    # Borrowed from https://github.com/nicklhy/AdversarialAutoEncoder/blob/master/data_factory.py#L40 (modified)
    def swiss_roll(self, batch_size, labels, n_classes):
        def sample(label, n_labels):
            uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(
                n_labels
            )
            r = sqrt(uni) * 3.0
            rad = np.pi * 4.0 * sqrt(uni)
            x = r * cos(rad)
            y = r * sin(rad)
            return np.array([x, y]).reshape((2,))

        dim_z = 2
        z = np.zeros((batch_size, dim_z), dtype=np.float32)
        for batch in range(batch_size):
            z[batch, :] = sample(labels[batch], n_classes)
        return z

    def get_prior(self, prior_type):
        if prior_type == "gaussian_mixture":
            return self.gaussian_mixture
        elif prior_type == "swiss_roll":
            return self.swiss_roll
        else:
            raise ValueError(
                "You passed in prior_type={}, supported types are: "
                "gaussian_mixture, swiss_roll".format(prior_type)
            )
