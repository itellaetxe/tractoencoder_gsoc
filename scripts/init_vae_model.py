import numpy as np
import tensorflow as tf
from tractoencoder_gsoc.models import vae_model

if __name__ == '__main__':
    # Example of using the model
    latent_space_dims = 32
    kernel_size = 3
    model = vae_model.IncrFeatStridedConvFCUpsampReflectPadVAE(latent_space_dims=latent_space_dims,
                                                               kernel_size=kernel_size)

    model.summary()

    # Example of using the model
    input_data = np.random.randn(1, 256, 3)
    # Duplicate the input_data along the batch axis 100 times
    input_data = tf.convert_to_tensor(np.tile(input_data, (1000, 1, 1)))

    output = model(input_data)
    print(output.shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000668))

    # Try the train method (x and y should be the same)
    model.fit(x=input_data, y=input_data, epochs=10, batch_size=100)
