import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow import keras
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import Print

from utils import draw_heatmap


class VizCallback(keras.callbacks.Callback):
    def __init__(self, model, data_generator, out_dir='model_select_viz'):
        self.model = model
        self.data_generator = data_generator
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.activation_maximization = ActivationMaximization(model,
                                                              self.softmax_to_linear,
                                                              clone=True,  # make sure to clone so currently training model does not replace activation!
                                                              )
        self.seed = tf.random.uniform((3, 224, 224, 3), 0, 255)

    def softmax_to_linear(self, m):
        m.layers[-1].activation = keras.activations.linear

    def on_epoch_end(self, epoch, logs=None):
        class_labels = self.data_generator.read_depth_labels
        n_classes = len(class_labels)
        imgs = []
        for i in range(n_classes):
            loss = lambda x: x[:, i]  # function that returns the class i value
            activation = self.activation_maximization(loss, steps=512)
            img = activation[0].astype(np.uint8)[..., 0]
            imgs.append(img)
        n_rows = math.ceil(math.sqrt(n_classes))
        n_cols = math.ceil(n_classes / n_rows)
        subplot_in = 2  # size in inches of each subplot
        fig = plt.figure(figsize=(n_cols * subplot_in, n_rows * subplot_in))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(n_rows, n_cols),  # creates grid of axes
                         axes_pad=0.2,  # pad between axes in inch.
                         )
        for i in range(n_classes):
            ax = grid[i]
            im = imgs[i]
            class_name = class_labels[i]
            #draw_heatmap(im, 0, ax=ax)
            ax.imshow(im, cmap='Reds')
            ax.set_title(class_name)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(os.path.join(self.out_dir, '%d.png' % epoch))
        plt.close()