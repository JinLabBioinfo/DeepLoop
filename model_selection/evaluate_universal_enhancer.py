import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # don't set up CUDA when generating docs by importing
import math
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

from utils import load_chr_ratio_matrix_from_sparse, sorted_nicely, draw_heatmap

parser = argparse.ArgumentParser()
parser.add_argument('--anchor_dir', required=True, type=str)
parser.add_argument('--matrix_size', required=False, type=int, default=256)
parser.add_argument('--step_size', required=False, type=int, default=512)
args = parser.parse_args()

test_dir = '../../../data/anchor_to_anchor/public_low_depth_tissue_data'
out_dir = 'universal_enhancer_heatmaps'
os.makedirs(out_dir, exist_ok=True)
n_examples = 2
anchor_dir = args.anchor_dir
matrix_size = args.matrix_size
step_size = args.step_size

model = keras.models.load_model('universal_enhancer.h5')

for tissue_dir in os.listdir(test_dir):
    chr_dir = os.path.join(test_dir, tissue_dir)
    x = []
    for chromosome in sorted_nicely(os.listdir(chr_dir)):
        chr_name = chromosome[:chromosome.find('.')]
        #print(chromosome, chr_name)
        sparse_matrix = load_chr_ratio_matrix_from_sparse(chr_dir, chromosome, anchor_dir=anchor_dir, chr_name=chr_name,  use_raw=False)
        indices = np.arange(0, sparse_matrix.shape[0] - matrix_size)
        for start_row in np.random.choice(indices, 4):
            rows = slice(start_row, start_row + matrix_size)
            tile = sparse_matrix[rows, rows].A
            fig, axs = plt.subplots(1, 2, figsize=(10, 6))
            draw_heatmap(tile, 0, ax=axs[0])
            axs[0].set_title('before enhance')

            tile = np.expand_dims(np.expand_dims(tile, -1), 0)
            pred = model.predict(tile)[0, ..., 0]
            pred = (pred + pred.T) * 0.5

            draw_heatmap(pred, 0, ax=axs[1])
            axs[1].set_title('universal enhance')

            fig.suptitle('%s-%s : %d - %d' % (tissue_dir, chr_name, start_row, start_row + matrix_size))
            fig.savefig(os.path.join(out_dir, '%s_%s_%d.png' % (tissue_dir, chr_name, start_row)))
            plt.close()