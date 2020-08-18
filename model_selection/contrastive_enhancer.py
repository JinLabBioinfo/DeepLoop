import os
import random
import argparse
import tensorboard
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras

from model_selection.downsample_data_generator import DataGenerator
from cnn_architectures import Autoencoder
from utils import load_chr_ratio_matrix_from_sparse, draw_heatmap


class UniversalEnhancer(keras.Model):

    def __init__(self, base_model):
        super(UniversalEnhancer, self).__init__()
        architectures = Autoencoder(matrix_size, step_size, start_filters=filters, filter_size=9,
                                    depth=depth)  # params from LoopEnhance
        self.enhance = architectures.get_unet_model()
        self.enhance.load_weights(base_model)
        latent_layer = self.enhance.get_layer(name='conv2d_10').output
        encoded_shape = latent_layer.get_shape().as_list()[1:]
        flat = keras.layers.Flatten()(latent_layer)
        self.encoder = keras.models.Model(self.enhance.input, flat)

    def call(self, inputs):
        x, y = inputs
        latent = self.encoder(x)
        latent_dot = tf.matmul(latent, tf.transpose(latent))
        latent_square_norm = tf.diag_part(latent_dot)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(latent_square_norm, 0) - 2.0 * latent_dot + tf.expand_dims(latent_square_norm, 1)
        contrastive_loss = tf.reduce_mean(distances)
        self.add_metric(contrastive_loss, 'distances')
        x_recon = self.enhance(x)
        recon_loss = keras.losses.mse(y, x_recon)
        self.add_metric(recon_loss, 'mse')
        loss = recon_loss + contrastive_loss
        self.add_loss(loss)
        return x_recon



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--target_dir', required=True, type=str)
parser.add_argument('--anchor_dir', required=True, type=str)
parser.add_argument('--matrix_size', required=False, type=int, default=128)
parser.add_argument('--step_size', required=False, type=int, default=64)
parser.add_argument('--depth', required=False, type=int, default=4)
parser.add_argument('--filters', required=False, type=int, default=4)
parser.add_argument('--n_epochs', required=False, type=int, default=20)
parser.add_argument('--pretrain', required=False, type=bool, default=False)
parser.add_argument('--base_model', required=False, type=str, default='enhance_CP_GZ_001_with_fake.h5')
args = parser.parse_args()

data_dir = args.data_dir
target_dir = args.target_dir
anchor_dir = args.anchor_dir
matrix_size = args.matrix_size
step_size = args.step_size
depth = args.depth
filters = args.filters
n_epochs = args.n_epochs
pretrain = args.pretrain
base_model = args.base_model

data_generator = DataGenerator(data_dir, target_dir, anchor_dir, matrix_size, step_size)
print('Input shape:', data_generator.input_shape)

univ_enhance = UniversalEnhancer(base_model)
univ_enhance.compile(optimizer=keras.optimizers.Adam())

# Define the Keras TensorBoard callback.
logdir = 'logs/'
os.makedirs(logdir, exist_ok=True)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)

univ_enhance.fit(data_generator.generate_batches(), epochs=n_epochs, verbose=1, callbacks=[tensorboard_callback], steps_per_epoch=data_generator.steps_per_epoch)