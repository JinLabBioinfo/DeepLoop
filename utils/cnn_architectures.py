import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout, Activation, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
import tensorflow as tf
from keras import backend as K


class Autoencoder:
    """
    Parent class for both denoise and enhance models.  Creates general model architectures and contains metric functions.
    """
    def __init__(self,
                 matrix_size=256,
                 step_size=64,
                 start_filters=4,
                 filter_size=3,
                 transpose_filter_size=2,
                 depth=0,
                 branching_rate=2,
                 dropout=0,
                 normalize=False,
                 activation='relu'):
        """
        Parent initializer for both denoise and enhance model.  Creates an autoencoder object with the given parameters.

        Args:
            matrix_size (:obj:`int`) : number of anchors to use for each input sample
            step_size (:obj:`int`) : size of steps used when generating batches.  Values less than ``matrix size`` will include overlapping regions
            start_filters (:obj:`int`) : number of filters in the first layer of the network
            filter_size (:obj:`int`): size of filters in each convolution layer, except the final linear convolution layer in the U-Net architecture
            depth (:obj:`int`) : number of layers in the U-Net model
            branching_rate (:obj:`int`) : multiplier for the number of filters in each successive layer of the U-net model
            dropout (:obj:`float`) : probability of randomly dropping certain network connection during training
            normalize (:obj:`bool`) : set to True to normalize inputs between ``[0, 1]`` before training
            activation (:obj:`str`) : activation function used after each convolutional layer
        """
        self.matrix_size = matrix_size
        self.step_size = step_size
        self.start_filters = start_filters
        self.filter_size = filter_size
        self.transpose_filter_size = transpose_filter_size
        self.depth = depth
        self.branching_rate = branching_rate
        self.dropout = dropout
        self.normalize = normalize
        self.activation = activation
        self.loss_plot = np.array([])
        self.loops_plot = np.array([])
        self.psnr_plot = np.array([])
        self.val_loss_plot = np.array([])
        self.val_psnr_plot = np.array([])
        self.val_jaccard_plot = np.array([])
        self.val_loops_plot = np.array([])
        self.val_percent_loops_plot = np.array([])

    def get_autoencoder(self, maxpool=True, upconv=False):
        """
        Builds a stacked convolutional autoencoder

        Args:
            maxpool (:obj:`bool`) : set to True to use map pooling to perform downsampling, otherwise use convolution layer with ``stride=2``
            upconv (:obj:`bool`) : set to True to upsample then apply convolution layer, otherwise use deconvolution (transpose convolution)

        Returns:
            Uncompiled Keras model
        """
        input_matrix = Input(shape=(None, None, 1))
        y = Conv2D(self.start_filters, self.filter_size, strides=1 if maxpool else 2, padding='same')(input_matrix)
        y = Activation(self.activation)(y)
        y = MaxPooling2D((2, 2))(y) if maxpool else y
        y = Conv2D(self.start_filters, self.filter_size, strides=1 if maxpool else 2, padding='same')(y)
        y = Activation(self.activation)(y)
        y = MaxPooling2D((2, 2))(y) if maxpool else y
        if upconv:
            y = UpSampling2D()(y)
            y = Conv2D(self.start_filters, self.filter_size, padding='same')(y)
            y = UpSampling2D()(y)
            y = Conv2D(self.start_filters, self.filter_size, padding='same')(y)
            y = Conv2D(1, self.filter_size, padding='same')(y)
            y = Activation(self.activation)(y)
        else:
            y = Conv2DTranspose(self.start_filters, self.transpose_filter_size, strides=2, padding='same')(y)
            y = Conv2DTranspose(self.start_filters, self.transpose_filter_size, strides=2, padding='same')(y)
            y = Conv2D(1, self.filter_size, padding='same')(y)
            y = Activation(self.activation)(y)
        model = Model(inputs=input_matrix, outputs=y)
        print(model.summary())
        return model

    @staticmethod
    def get_hi_c_plus():
        input_matrix = Input(shape=(None, None, 1))
        y = Conv2D(8, 9, padding='same')(input_matrix)
        y = Activation('relu')(y)
        y = Conv2D(8, 1, padding='same')(y)
        y = Activation('relu')(y)
        y = Conv2D(1, 5, padding='same')(y)
        y = Activation('relu')(y)
        model = Model(inputs=input_matrix, outputs=y)
        print(model.summary())
        return model

    # UNet: code based on https://github.com/pietz/unet-keras
    def get_unet_model(self, batch_norm=False, maxpool=True, upconv=True, residual=True, pad_pooling=True):
        """
        Builds a U-Net model

        Args:
            batch_norm (:obj:`bool`) : set to True to use batch normalization after each convolution layer
            maxpool (:obj:`bool`) : set to True to use map pooling to perform downsampling, otherwise use convolution layer with ``stride=2``
            upconv (:obj:`bool`) : set to True to upsample then apply convolution layer, otherwise use deconvolution (transpose convolution)
            residual (:obj:`bool`) : set to True to concatenate feature maps from contracting path to expansion path

        Returns:
            Uncompiled Keras model
        """

        def conv_block(x, num_filters, acti, bn, res, do=0):
            y = Conv2D(num_filters, self.filter_size, padding='same')(x)
            y = Activation(acti)(y)
            y = BatchNormalization()(y) if bn else y
            y = Dropout(do)(y) if do > 0 else y
            y = Conv2D(num_filters, self.filter_size, padding='same')(y)
            y = Activation(acti)(y)
            y = BatchNormalization()(y) if bn else y

            return Concatenate()([x, y]) if res else y

        def level_block(x, num_filters, depth, branch_rate, acti, dropout, bn, max_pool, up_sample, res):
            if depth > 0:
                y = conv_block(x, num_filters, acti, bn, res)
                padding = 'same' if pad_pooling else 'valid'
                x = MaxPooling2D(padding=padding)(y) if max_pool else Conv2D(num_filters, self.filter_size, strides=2, padding='same')(y)
                x = level_block(x, int(branch_rate * num_filters), depth - 1, branch_rate, acti, dropout, bn, max_pool,
                                up_sample, res)
                if up_sample:
                    x = UpSampling2D()(x)
                    x = Conv2D(num_filters, self.filter_size, padding='same')(x)
                    x = Activation(acti)(x)
                else:
                    x = Conv2DTranspose(num_filters, self.filter_size, strides=2, padding='same')(x)
                    x = Activation(acti)(x)
                y = Concatenate()([y, x])
                x = conv_block(y, num_filters, acti, bn, res)
            else:
                x = conv_block(x, num_filters, acti, bn, res, dropout)

            return x

        input_matrix = Input(shape=(None, None, 1))
        output = level_block(input_matrix, self.start_filters, self.depth, self.branching_rate, self.activation,
                             self.dropout, batch_norm, maxpool, upconv, residual)

        output = Conv2D(1, 1)(output)

        model = Model(inputs=input_matrix, outputs=output)
        print(model.summary())
        return model

    def get_latent_unet_model(self, batch_norm=False, maxpool=True, upconv=True, residual=True):
        """U-Net Generator"""

        self.gf = self.start_filters

        def conv2d(layer_input, filters, f_size=9, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=self.filter_size, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=(self.matrix_size, self.matrix_size, 1))

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        #d6 = conv2d(d5, self.gf * 8)
        #d7 = conv2d(d6, self.gf * 8)

        fc = Dense(128)(d5)

        # Upsampling
        #u1 = deconv2d(fc, d6, self.gf * 8)
        #u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(fc, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(1, kernel_size=1, strides=1)(u7)

        return Model(d0, output_img)

    def model_params_to_name(self):
        """
        Converts the parameters of the model into a string for easily organizing saved models.

        Returns:
            `str` : string of underscore separated parameters
        """
        model_params = '_' + str(self.start_filters) + '_' + str(self.filter_size) + '_' + str(self.depth)
        return model_params

    @staticmethod
    def tf_log10(x):
        """
        Take the base 10 logarithm of a tensor

        Args:
            x (:obj:`tensorflow.Tensor`) : tensor object

        Returns:
            Base 10 logarithm of input tensor
        """
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def psnr(self, y_true, y_pred):
        """
        Compute the peak signal to noise ratio (PSNR) between the target and predicted matrices

        Args:
            y_true (:obj:`tensorflow.Tensor`) : training target batch
            y_pred (:obj:`tensorflow.Tensor`) : predicted model output batch

        Returns:
            PSNR between ``y_true`` and ``y_pred`` as a tensor
        """
        if self.normalize:
            max_pixel = 1.0
        else:
            max_pixel = 5.0
        y_pred = K.clip(y_pred, 0.0, max_pixel)
        return 10.0 * self.tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

    @staticmethod
    def jaccard_metric_weighted(smooth=100):
        """
        Compute the weighted jaccard distance between training target and prediction

        Args:
            smooth: smoothing factor used to avoid NaN values when dividing intersection by union

        Returns:
            Weighted jaccard distance as a tensor
        """
        def jaccard_distance(y_true, y_pred):
            """
            Ref: https://en.wikipedia.org/wiki/Jaccard_index
            """
            mask_true = tf.not_equal(y_true, 0)
            mask_pred = tf.not_equal(y_pred, 0)
            mask = tf.logical_and(mask_true, mask_pred)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
            intersection = K.sum(K.minimum(y_pred, y_true))
            union = K.sum(K.maximum(y_pred, y_true))
            jac = 1 - (intersection + smooth) / (union + smooth)
            return jac

        return jaccard_distance