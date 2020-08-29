import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow import keras
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import Print

from utils import draw_heatmap, load_chr_ratio_matrix_from_sparse


class VizCallback(keras.callbacks.Callback):
    def __init__(self, model, data_generator, out_dir='model_select_viz'):
        self.model = model
        self.data_generator = data_generator
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        fig, axs = plt.subplots(2, len(self.data_generator.depth_dirs) + 1,
                                figsize=(6 * len(self.data_generator.depth_dirs) + 2, 6 * 2))
        # example_chr = data_generator.current_chr
        # example_loc = random.randint(0, data_generator.chr_lengths[example_chr] - matrix_size)
        example_chr = 'chr6'
        example_loc = 16193
        matrix_size = self.data_generator.matrix_size
        target_dir = self.data_generator.target_dir
        anchor_dir = self.data_generator.anchor_dir
        rows = slice(example_loc, example_loc + matrix_size)
        target_file = self.data_generator.get_chromosome_file(target_dir, example_chr)
        target_matrix = load_chr_ratio_matrix_from_sparse(target_dir, target_file, anchor_dir)
        target_tile = target_matrix[rows, rows].A
        for plot_i, (depth_dir, read_depth) in enumerate(
                zip(self.data_generator.depth_dirs, self.data_generator.read_depth_labels)):
            rep_dirs = os.listdir(depth_dir)
            for f in rep_dirs:
                if 'summary' in f:
                    rep_dirs.remove(f)
                    break
            genome = os.path.join(depth_dir, rep_dirs[0])
            chr_file = self.data_generator.get_chromosome_file(genome, example_chr)
            matrix = load_chr_ratio_matrix_from_sparse(genome, chr_file, anchor_dir)
            tile = matrix[rows, rows].A
            draw_heatmap(tile, 0, ax=axs[0][plot_i])
            #axs[0][plot_i].imshow(tile, cmap='Reds')
            axs[0][plot_i].set_title(read_depth)
            axs[0][plot_i].set_xticks([])
            axs[0][plot_i].set_yticks([])

            tile = np.expand_dims(tile, -1)
            tile = np.expand_dims(tile, 0)

            enhanced = self.model.enhance.predict(tile)[0, ..., 0]
            enhanced = (enhanced + enhanced.T) * 0.5

            draw_heatmap(enhanced, 0, ax=axs[1][plot_i])
            #axs[1][plot_i].imshow(enhanced, cmap='Reds')
            axs[1][plot_i].set_xticks([])
            axs[1][plot_i].set_yticks([])

        draw_heatmap(target_tile, 0, ax=axs[1][-1])
        axs[1][-1].set_title('Target')
        axs[1][-1].set_xticks([])
        axs[1][-1].set_yticks([])


        fig.savefig('enhance_training/%d.png' % (epoch))
        plt.close()
        self.model.enhance.save('models/universal_enhancer_%d.h5' % epoch)