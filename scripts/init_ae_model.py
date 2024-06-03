import numpy as np
import tensorflow as tf

from tractoencoder_gsoc import ae_model

if __name__ == '__main__':
    # Example of using the model
    latent_space_dims = 32
    kernel_size = 3
    model = ae_model.IncrFeatStridedConvFCUpsampReflectPadAE(latent_space_dims=latent_space_dims,
                                                             kernel_size=kernel_size)

    model.summary()

    # Example of using the model
    input_data = np.random.randn(100, 256, 3)

    output = model(input_data)
    print(output.shape)

    # Try the train method
    model.fit(x=input_data, y=input_data, epochs=10, batch_size=5)
