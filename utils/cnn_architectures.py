import numpy as np
import tensorflow as tf
from tensorflow import keras

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
        input_matrix = keras.layers.Input(shape=(None, None, 1))
        y = keras.layers.Conv2D(self.start_filters, self.filter_size, strides=1 if maxpool else 2, padding='same')(input_matrix)
        y = keras.layers.Activation(self.activation)(y)
        y = keras.layers.MaxPooling2D((2, 2))(y) if maxpool else y
        y = keras.layers.Conv2D(self.start_filters, self.filter_size, strides=1 if maxpool else 2, padding='same')(y)
        y = keras.layers.Activation(self.activation)(y)
        y = keras.layers.MaxPooling2D((2, 2))(y) if maxpool else y
        if upconv:
            y = keras.layers.UpSampling2D()(y)
            y = keras.layers.Conv2D(self.start_filters, self.filter_size, padding='same')(y)
            y = keras.layers.UpSampling2D()(y)
            y = keras.layers.Conv2D(self.start_filters, self.filter_size, padding='same')(y)
            y = keras.layers.Conv2D(1, self.filter_size, padding='same')(y)
            y = keras.layers.Activation(self.activation)(y)
        else:
            y = keras.layers.Conv2DTranspose(self.start_filters, self.transpose_filter_size, strides=2, padding='same')(y)
            y = keras.layers.Conv2DTranspose(self.start_filters, self.transpose_filter_size, strides=2, padding='same')(y)
            y = keras.layers.Conv2D(1, self.filter_size, padding='same')(y)
            y = keras.layers.Activation(self.activation)(y)
        model = keras.models.Model(inputs=input_matrix, outputs=y)
        print(model.summary())
        return model

    @staticmethod
    def get_hi_c_plus():
        input_matrix = keras.layers.Input(shape=(None, None, 1))
        y = keras.layers.Conv2D(8, 9, padding='same')(input_matrix)
        y = keras.layers.Activation('relu')(y)
        y = keras.layers.Conv2D(8, 1, padding='same')(y)
        y = keras.layers.Activation('relu')(y)
        y = keras.layers.Conv2D(1, 5, padding='same')(y)
        y = keras.layers.Activation('relu')(y)
        model = keras.models.Model(inputs=input_matrix, outputs=y)
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
            y = keras.layers.Conv2D(num_filters, self.filter_size, padding='same')(x)
            y = keras.layers.Activation(acti)(y)
            y = keras.layers.BatchNormalization()(y) if bn else y
            y = keras.layers.Dropout(do)(y) if do > 0 else y
            y = keras.layers.Conv2D(num_filters, self.filter_size, padding='same')(y)
            y = keras.layers.Activation(acti)(y)
            y = keras.layers.BatchNormalization()(y) if bn else y

            return keras.layers.Concatenate()([x, y]) if res else y

        def level_block(x, num_filters, depth, branch_rate, acti, dropout, bn, max_pool, up_sample, res):
            if depth > 0:
                y = conv_block(x, num_filters, acti, bn, res)
                padding = 'same' if pad_pooling else 'valid'
                x = keras.layers.MaxPooling2D(padding=padding)(y) if max_pool else keras.layers.Conv2D(num_filters,
                                                                                                       self.filter_size,
                                                                                                       strides=2,
                                                                                                       padding='same')(y)
                x = level_block(x, int(branch_rate * num_filters), depth - 1, branch_rate, acti, dropout, bn, max_pool,
                                up_sample, res)
                if up_sample:
                    x = keras.layers.UpSampling2D()(x)
                    x = keras.layers.Conv2D(num_filters, self.filter_size, padding='same')(x)
                    x = keras.layers.Activation(acti)(x)
                else:
                    x = keras.layers.Conv2DTranspose(num_filters, self.filter_size, strides=2, padding='same')(x)
                    x = keras.layers.Activation(acti)(x)
                y = keras.layers.Concatenate()([y, x])
                x = conv_block(y, num_filters, acti, bn, res)
            else:
                x = conv_block(x, num_filters, acti, bn, res, dropout)

            return x

        input_matrix = keras.layers.Input(shape=(self.matrix_size, self.matrix_size, 1))
        output = level_block(input_matrix, self.start_filters, self.depth, self.branching_rate, self.activation,
                             self.dropout, batch_norm, maxpool, upconv, residual)

        output = keras.layers.Conv2D(1, 1)(output)

        model = keras.models.Model(inputs=input_matrix, outputs=output)
        print(model.summary())
        return model


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
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
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
        y_pred = keras.backend.clip(y_pred, 0.0, max_pixel)
        return 10.0 * self.tf_log10((max_pixel ** 2) / (keras.backend.mean(keras.backend.square(y_pred - y_true))))

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
            intersection = keras.backend.sum(keras.backend.minimum(y_pred, y_true))
            union = keras.backend.sum(keras.backend.maximum(y_pred, y_true))
            jac = 1 - (intersection + smooth) / (union + smooth)
            return jac

        return jaccard_distance
