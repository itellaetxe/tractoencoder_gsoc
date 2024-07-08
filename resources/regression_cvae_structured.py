# Taken from: https://github.com/QingyuZhao/VAE-for-Regression/blob/master/3D_MRI_VAE_regression.py


##
# Usage: python 3D_MRI_VAE_regression.py ROI_x ROI_y ROI_z Size_x Size_y Size_z
# ROI_x,y,z, Size_x,y,z: Selecting a specific ROI box for analysis
# Reach out to http://cnslab.stanford.edu/ for data usage

from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from keras import layers, Layer, Model, Sequential, losses, constraints, Input, saving
from keras import regularizers
from keras import backend as K

from sklearn.model_selection import StratifiedKFold
import numpy as np
import nibabel as nib
import scipy as sp
import scipy.ndimage
from sklearn.metrics import mean_squared_error, r2_score

from tractoencoder_gsoc.utils import pre_pad

# Define Encoder (will take (n, 256, 3) shaped tf.Tensor)
class Encoder(Layer):
    def __init__(self,
                 latent_space_dims: int = 32,
                 kernel_size: int = 32,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.latent_space_dims = latent_space_dims
        self.kernel_size = kernel_size

        self.encod_conv1 = pre_pad(
            layers.Conv1D(16, self.kernel_size, strides=2, padding="valid",
                          name="encoder_conv1",
                          activation="relu")
        )
        self.encod_max_pool1 = layers.MaxPooling1D(pool_size=2,
                                                   name="encoder_max_pool1")

        self.encod_conv2 = pre_pad(
            layers.Conv1D(32, self.kernel_size, strides=2, padding="valid",
                          name="encoder_conv2",
                          activation="relu")
        )
        self.encod_max_pool2 = layers.MaxPooling1D(pool_size=2,
                                                   name="encoder_max_pool2")

        self.encod_conv3 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=2, padding="valid",
                          name="encoder_conv3",
                          activation="relu")
        )
        self.encod_max_pool3 = layers.MaxPooling1D(pool_size=2,
                                                   name="encoder_max_pool3")

        self.encod_flatten = layers.Flatten(name="encoder_flatten")
        self.encod_dropout = layers.Dropout(0.3, name="encoder_dropout")

        self.encod_dense1 = layers.Dense(4 * self.latent_space_dims,
                                         name="encoder_dense1",
                                         activation="tanh",
                                         kernel_regularizer=regularizers.l2(l2=0.01))

        self.encod_z_mean_feature = layers.Dense(2 * self.latent_space_dims,
                                                 name="encoder_z_mean_feature",
                                                 activation="tanh")
        self.encod_z_mean = layers.Dense(self.latent_space_dims,
                                         name="encoder_z_mean",
                                         activation="tanh")

        self.encod_z_log_var_feature = layers.Dense(2 * self.latent_space_dims,
                                                    name="encoder_log_var_feature",
                                                    activation="tanh")
        self.encod_z_log_var = layers.Dense(self.latent_space_dims,
                                            name="encoder_z_lov_var")

        self.encod_r_mean_feature = layers.Dense(2 * self.latent_space_dims,
                                                 name="encoder_r_mean_feature",
                                                 activation="tanh")
        self.encod_r_mean = layers.Dense(1, name="encoder_r_mean")

        self.encod_r_log_var_feature = layers.Dense(2 * self.latent_space_dims,
                                                    name="encoder_r_log_var_feature",
                                                    activation="tanh")
        self.encod_r_log_var = layers.Dense(1, name="encoder_r_log_var")

        self.encod_z = layers.Lambda(sampling,
                                     output_shape=(self.latent_space_dims,),
                                     name="z")
        self.encod_r = layers.Lambda(sampling,
                                     output_shape=(1,),
                                     name="r")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_space_dims": saving.serialize_keras_object(self.latent_space_dims),
            "kernel_size": saving.serialize_keras_object(self.kernel_size)
        }
        return {**base_config, **config}

    def from_config(self):
        # TODO: Implement this to deserialize the model properly
        pass

    def call(self, input_data: tf.Tensor, input_age: int):
        x = input_data
        age = input_age

        h1 = self.encod_conv1(x)
        h2 = self.encod_max_pool1(h1)
        h3 = self.encod_conv2(h2)
        h4 = self.encod_max_pool2(h3)
        h5 = self.encod_conv3(h4)
        h6 = self.encod_max_pool3(h5)

        h7 = self.encod_flatten(h6)
        h8 = self.encod_dropout(h7)

        h_dense = self.encod_dense1(h8)

        h_feature_z_mean = self.encod_z_mean_feature(h_dense)
        h_z_mean = self.encod_z_mean(h_feature_z_mean)
        h_feature_z_log_var = self.encod_z_log_var_feature(h_dense)
        h_z_log_var = self.encod_z_log_var(h_feature_z_log_var)

        h_feature_r_mean = self.encod_r_mean_feature(h_dense)
        h_r_mean = self.encod_r_mean(h_feature_r_mean)
        h_feature_r_log_var = self.encod_r_log_var_feature(h_dense)
        h_r_log_var = self.encod_r_log_var(h_feature_r_log_var)

        h_z = self.encod_z([h_z_mean, h_z_log_var])
        h_r = self.encod_r([h_r_mean, h_r_log_var])

        return h_z, h_r


class Generator(Layer):
    def __init__(self, latent_space_dims, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.latent_space_dims = latent_space_dims
        self.pz_mean = layers.Dense(self.latent_space_dims,
                                    name="pz_mean",
                                    kernel_constraint=constraints.UnitNorm())
        self.pz_log_var = layers.Dense(1, name="pz_log_var",
                                       kernel_constraint=constraints.MaxNorm(0))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_space_dims": saving.serialize_keras_object(self.latent_space_dims)
        }
        return {**base_config, **config}

    def from_config(self):
        # TODO: Implement this to deserialize the model properly
        pass

    def call(self, input_data):
        pz_mean = self.pz_mean(input_data)
        pz_log_var = self.pz_log_var(input_data)
        return pz_mean, pz_log_var


class Decoder(Layer):
    def __init__(self, latent_space_dims, kernel_size=3, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.encoder_out_size = latent_space_dims
        self.kernel_size = kernel_size

        self.decod_dense1 = layers.Dense(2 * self.encoder_out_size,
                                         name="decoder_dense1",
                                         activation="tanh",
                                         kernel_regularizer=regularizers.l2(l2=0.01))
        self.decod_dense2 = layers.Dense(4 * self.encoder_out_size,
                                         name="decoder_dense2",
                                         activation="tanh",
                                         kernel_regularizer=regularizers.l2(l2=0.01))
        self.decod_dense3 = layers.Dense(int((256 / 8) * 16 * 4),
                                         name="decoder_dense3",
                                         activation="relu",
                                         kernel_regularizer=regularizers.l2(l2=0.01))
        self.decod_reshape = layers.Reshape((int((256 / 8) * 16 * 4), 1),
                                            name="decoder_reshape")

        self.decod_conv1 = layers.Conv1D(64, kernel_size=self.kernel_size,
                                         padding="valid", name="encoder_conv1",
                                         activation="relu")
        self.decod_upsampl1 = layers.UpSampling1D(size=2,
                                                  name="decoder_upsampling1")

        self.decod_conv2 = layers.Conv1D(32, kernel_size=self.kernel_size,
                                         padding="valid", name="encoder_conv2",
                                         activation="relu")
        self.decod_upsampl2 = layers.UpSampling1D(size=2,
                                                  name="decoder_upsampling2")

        self.decod_conv3 = layers.Conv1D(16, kernel_size=self.kernel_size,
                                         padding="valid", name="encoder_conv3",
                                         activation="relu")
        self.decod_upsampl3 = layers.UpSampling1D(size=2,
                                                  name="decoder_upsampling3")


def init_model(latent_space_dims=32, kernel_size=3):
    input_data = Input(shape=(256, 3), name='input_streamline')

    # encode
    encoder = Encoder(latent_space_dims=latent_space_dims,
                      kernel_size=kernel_size)
    encoded = encoder(input_data)

    # decode
    decoder = Decoder(encoder.encoder_out_size,
                      kernel_size=kernel_size)
    decoded = decoder(encoded)
    output_data = decoded

    # Instantiate model and name it
    model = Model(input_data, output_data)
    model.name = 'RegressionVAE'
    return model


class RegressionVAE():
    def __init__(self, latent_space_dims=32, kernel_size=3):

        # Parameter initialization
        self.latent_space_dims = latent_space_dims
        self.kernel_size = kernel_size
        self.input = Input(shape=(256, 3), name="input_streamlines")

        self.model = init_model(latent_space_dims=self.latent_space_dims,
                                kernel_size=self.kernel_size)

    def __call__(self, x):
        return self.model(x)

    def compile(self, **kwargs):
        """
        Configure the model for training
        """
        kwargs['optimizer'].weight_decay = 0.13
        self.model.compile(**kwargs)

    def summary(self, **kwargs):
        """
        Get the summary of the model.
        # TODO: Complete docstring
        The summary is textual and includes information about:
        The layers and their order in the model.
        The output shape of each layer.
        """
        return self.model.summary(**kwargs)

    def fit(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        Args:
            x (_type_): _description_
            y (_type_): _description_
            batch_size (_type_, optional): _description_. Defaults to None.
            epochs (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        if isinstance(kwargs['x'], nib.streamlines.ArraySequence):
            kwargs['x'] = np.array(kwargs['x'])
        if isinstance(kwargs['y'], nib.streamlines.ArraySequence):
            kwargs['y'] = np.array(kwargs['y'])
        return self.model.fit(*args, **kwargs)
        # TODO (perhaps): write train loop manually?

    def save_weights(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        self.model.save_weights(*args, **kwargs)

    def save(self, *args, **kwargs):
        """_summary_
        # TODO: Complete docstring
        """
        self.model.save(*args, **kwargs)


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tf.Tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tf.Tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    K.random_uniform(shape=(batch, 1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def augment_by_transformation(data, age, n):

    if n <= data.shape[0]:
        return data
    else:
        raw_n = data.shape[0]
        m = n - raw_n
        for i in range(0, m):
            new_data = np.zeros((1, data.shape[1],
                                 data.shape[2],
                                 data.shape[3], 1))
            idx = np.random.randint(0, raw_n)
            new_age = age[idx]
            new_data[0] = data[idx].copy()
            new_data[0, :, :, :, 0] = sp.ndimage.interpolation.rotate(new_data[0, :, :, :, 0],
                                                                      np.random.uniform(-1, 1),
                                                                      axes=(1, 0),
                                                                      reshape=False)
            new_data[0, :, :, :, 0] = sp.ndimage.interpolation.rotate(new_data[0, :, :, :, 0],
                                                                      np.random.uniform(-1, 1),
                                                                      axes=(0, 1),
                                                                      reshape=False)
            new_data[0, :, :, :, 0] = sp.ndimage.shift(new_data[0, :, :, :, 0],
                                                       np.random.uniform(-1, 1))
            data = np.concatenate((data, new_data), axis=0)
            age = np.append(age, new_age)

        return data, age

def augment_by_noise(data, n, sigma):
    if n <= data.shape[0]:
        return data
    else:
        m = n - data.shape[0]
        for i in range(0, m):
            new_data = np.zeros((1, data.shape[1], data.shape[2], data.shape[3], 1))
            new_data[0] = data[np.random.randint(0, data.shape[0])]
            noise = np.clip(np.random.normal(0, sigma, (data.shape[1], data.shape[2], data.shape[3], 1)), -3 * sigma, 3 * sigma)
            new_data[0] += noise
            data = np.concatenate((data, new_data), axis=0)
        return data


def augment_by_flip(data):
    data_flip = np.flip(data, 1)
    data = np.concatenate((data, data_flip), axis=0)
    return data


if __name__ == "__main__":
    # Main Script #######
    # min_x = int(sys.argv[1])
    # min_y = int(sys.argv[2])
    # min_z = int(sys.argv[3])
    # patch_x = int(sys.argv[4])
    # patch_y = int(sys.argv[5])
    # patch_z = int(sys.argv[6])

    # dropout_alpha = float(sys.argv[7])
    # L2_reg = float(sys.argv[8])

    # CNN Parameters
    # dropout_alpha = 0.5
    ft_bank_baseline = 16
    latent_dim = 16
    augment_size = 1000
    # L2_reg= 0.00
    binary_image = False


    # load data
    file_idx = np.loadtxt('./access.txt')
    age = np.loadtxt('./age.txt')
    subject_num = file_idx.shape[0]

    data = np.zeros((subject_num, patch_x, patch_y, patch_z, 1))
    i = 0
    for subject_idx in file_idx:
        subject_string = format(int(subject_idx), '04d')
        filename_full = '/fs/neurosci01/qingyuz/lab_structural/affine_2mm/' + subject_string + '_baseline.nii.gz'

        img = nib.load(filename_full)
        img_data = img.get_fdata()

        data[i, :, :, :, 0] = img_data[min_x: min_x + patch_x, min_y: min_y + patch_y, min_z: min_z + patch_z]
        data[i, :, :, :, 0] = (data[i, :, :, :, 0] - np.mean(data[i, :, :, :, 0])) / np.std(data[i, :, :, :, 0])

        # output an example
        array_img = nib.Nifti1Image(np.squeeze(data[i, :, :, :, 0]), np.diag([1, 1, 1, 1]))
        filename = 'processed_example.nii.gz'
        nib.save(array_img, filename)

        i += 1


    # Cross Validation
    print("Data size \n", data.shape)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    fake = np.zeros((data.shape[0]))
    pred = np.zeros((age.shape))

    for train_idx, test_idx in skf.split(data, fake):

        train_data = data[train_idx]
        train_age = age[train_idx]

        test_data = data[test_idx]
        test_age = age[test_idx]

        # build encoder model
        input_r = Input(shape=(1, ), name='ground_truth')
        input_image = Input(shape=(patch_x, patch_y, patch_z, 1), name='input_image')
        feature = layers.Conv3D(ft_bank_baseline, activation='relu', kernel_size=(3, 3, 3), padding='same')(input_image)
        feature = layers.MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = layers.Conv3D(ft_bank_baseline * 2, activation='relu', kernel_size=(3, 3, 3), padding='same')(feature)
        feature = layers.MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = layers.Conv3D(ft_bank_baseline * 4, activation='relu', kernel_size=(3, 3, 3), padding='same')(feature)
        feature = layers.MaxPooling3D(pool_size=(2, 2, 2))(feature)

        feature = layers.Flatten()(feature)
        feature = layers.Dropout(dropout_alpha)(feature)
        feature_dense = layers.Dense(latent_dim * 4, activation='tanh', kernel_regularizer=regularizers.l2(l2=L2_reg))(feature)

        feature_z_mean = layers.Dense(latent_dim * 2, activation='tanh')(feature_dense)
        z_mean = layers.Dense(latent_dim, name='z_mean')(feature_z_mean)
        feature_z_log_var = layers.Dense(latent_dim * 2, activation='tanh')(feature_dense)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(feature_z_log_var)

        feature_r_mean = layers.Dense(latent_dim * 2, activation='tanh')(feature_dense)
        r_mean = layers.Dense(1, name='r_mean')(feature_r_mean)
        feature_r_log_var = layers.Dense(latent_dim * 2, activation='tanh')(feature_dense)
        r_log_var = layers.Dense(1, name='r_log_var')(feature_r_log_var)

        # use reparameterization trick to push the sampling out as input
        z_mondongo = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        r = layers.Lambda(sampling, output_shape=(1,), name='r')([r_mean, r_log_var])

        # instantiate encoder model
        encoder = Model([input_image,input_r], [z_mean, z_log_var, z_mondongo, r_mean, r_log_var, r], name='encoder')
        encoder.summary()

        # build generator model
        generator_input = Input(shape=(1,), name='genrator_input')
        # inter_z_1 = layers.Dense(int(latent_dim/4), activation='tanh', kernel_constraint=constraints.UnitNorm(), name='inter_z_1')(generator_input)
        # inter_z_2 = layers.Dense(int(latent_dim/2), activation='tanh', kernel_constraint=constraints.UnitNorm(), name='inter_z_2')(inter_z_1)
        # pz_mean = layers.Dense(latent_dim, name='pz_mean')(inter_z_2)
        pz_mean = layers.Dense(latent_dim, name='pz_mean', kernel_constraint=constraints.UnitNorm())(generator_input)
        pz_log_var = layers.Dense(1, name='pz_log_var',kernel_constraint=constraints.MaxNorm(0))(generator_input)


        # instantiate generator model
        generator = Model(generator_input, [pz_mean,pz_log_var], name='generator')
        generator.summary()    

        # build decoder model
        latent_input = Input(shape=(latent_dim,), name='z_sampling')
        decoded = layers.Dense(latent_dim*2, activation='tanh',kernel_regularizer=regularizers.l2(l2=L2_reg))(latent_input)
        decoded = layers.Dense(latent_dim*4, activation='tanh',kernel_regularizer=regularizers.l2(l2=L2_reg))(decoded)
        decoded = layers.Dense(int(patch_x/8*patch_y/8*patch_z/8*ft_bank_baseline*4), activation='relu',kernel_regularizer=regularizers.l2(l2=L2_reg))(decoded)
        decoded = layers.Reshape((int(patch_x/8),int(patch_y/8),int(patch_z/8),ft_bank_baseline*4))(decoded)
        
        decoded = layers.Conv3D(ft_bank_baseline*4, kernel_size=(3, 3, 3),padding='same')(decoded)
        decoded = layers.Activation('relu')(decoded)
        decoded = layers.UpSampling3D((2,2,2))(decoded)

        decoded = layers.Conv3D(ft_bank_baseline*2, kernel_size=(3, 3, 3),padding='same')(decoded)
        decoded = layers.Activation('relu')(decoded)
        decoded = layers.UpSampling3D((2,2,2))(decoded)

        decoded = layers.Conv3D(ft_bank_baseline, kernel_size=(3, 3, 3),padding='same')(decoded)
        decoded = layers.Activation('relu')(decoded)
        decoded = layers.UpSampling3D((2,2,2))(decoded)

        decoded = layers.Conv3D(1, kernel_size=(3, 3, 3),padding='same')(decoded)    
        if binary_image:
            outputs = layers.Activation('sigmoid')(decoded)
        else:
            outputs = decoded

        # instantiate decoder model
        decoder = Model(latent_input, outputs, name='decoder')
        decoder.summary()
        
        # instantiate VAE model
        pz_mean,pz_log_var = generator(encoder([input_image,input_r])[5])
        outputs = decoder(encoder([input_image,input_r])[2])
        vae = Model([input_image,input_r], [outputs, pz_mean,pz_log_var], name='vae_mlp')

        
        if binary_image:
            reconstruction_loss = K.mean(losses.BinaryCrossentropy(input_image,outputs), axis=[1,2,3])
        else:
            reconstruction_loss = K.mean(losses.MeanAbsoluteError(input_image,outputs), axis=[1,2,3])

        kl_loss = 1 + z_log_var - pz_log_var - K.tf.divide(K.square(z_mean-pz_mean),K.exp(pz_log_var)) - K.tf.divide(K.exp(z_log_var),K.exp(pz_log_var))
        kl_loss = -0.5*K.sum(kl_loss, axis=-1)
        label_loss = K.tf.divide(0.5*K.square(r_mean - input_r), K.exp(r_log_var)) +  0.5 * r_log_var

        vae_loss = K.mean(reconstruction_loss+kl_loss+label_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()
    
        #break
        # augment data
        train_data_aug,train_age_aug = augment_by_transformation(train_data,train_age,augment_size)
        print("Train data shape: ",train_data_aug.shape)

        # training
        vae.fit([train_data_aug,train_age_aug],
                verbose=2,
                batch_size=64,
                epochs = 80)

        vae.save_weights('vae_weights.h5')
        encoder.save_weights('encoder_weights.h5')
        generator.save_weights('generator_weights.h5')
        decoder.save_weights('decoder_weights.h5')

        # testing
        [z_mean, z_log_var, z_mondongo, r_mean, r_log_var, r_vae] = encoder.predict([test_data,test_age],batch_size=64)
        pred[test_idx] = r_mean[:,0]

        filename = 'prediction_'+str(dropout_alpha)+'_'+str(L2_reg)+'.npy'
        np.save(filename,pred)

    ## CC accuracy
    print("MSE: ",mean_squared_error(age,pred))
    print("R2: ",r2_score(age, pred))


    ## Training on all data to learn a mega generative model
    train_data_aug,train_age_aug = augment_by_transformation(data,age,augment_size)
    vae.fit([data,age],
            verbose=2,
            batch_size=64,
            epochs = 80)

    ## Sample from latent space for visualizing the aging brain
    #generator.load_weights('generator_weights.h5')
    #decoder.load_weights('decoder_weights.h5')
    # this range depends on the resulting encoded latent space
    r = [-2, -1.5, -1, -0.5, 0, 1, 1.5, 2.5, 3.5, 4.5]

    pz_mean = generator.predict(r,batch_size=64)
    outputs = decoder.predict(pz_mean,batch_size=64)

    for i in range(0,10):   
        array_img = nib.Nifti1Image(np.squeeze(outputs[i,:,:,:,0]),np.diag([1, 1, 1, 1]))
        
        filename = 'generated'+str(i)+'.nii.gz'
        nib.save(array_img,filename)

    exit()
