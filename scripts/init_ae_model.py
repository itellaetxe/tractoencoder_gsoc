import numpy as np

from tractoencoder_gsoc import ae_model

if __name__ == '__main__':
    # Example of using the model
    latent_space_dims = 32
    kernel_size = 3
    model = ae_model.IncrFeatStridedConvFCUpsampReflectPadAE(latent_space_dims=latent_space_dims,
                                                             kernel_size=kernel_size)

    model.summary()

    # Example of using the model
    input_data = np.random.randn(1, 256, 3)
    # Duplicate the input_data along the batch axis 100 times
    input_data = np.tile(input_data, (50, 1, 1))

    output = model(input_data)
    model.model.compile()
    print(output.shape)

    # Try the train method (x and y should be the same)
    model.fit(x=input_data, y=input_data, epochs=200, batch_size=20)
