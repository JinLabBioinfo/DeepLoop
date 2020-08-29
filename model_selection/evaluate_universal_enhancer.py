import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # don't set up CUDA when generating docs by importing
import math
import time
import random
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

from utils import load_chr_ratio_matrix_from_sparse, sorted_nicely, draw_heatmap


def get_chromosome_file(dir_name, chr_name):
    a2a_files = os.listdir(dir_name)
    for file in a2a_files:
        if chr_name + '.' in file or file.endswith(chr_name):
            return file


parser = argparse.ArgumentParser()
parser.add_argument('--anchor_dir', required=True, type=str)
parser.add_argument('--matrix_size', required=False, type=int, default=128)
parser.add_argument('--step_size', required=False, type=int, default=512)
args = parser.parse_args()

test_dir = '../../../data/anchor_to_anchor/public_low_depth_tissue_data'
enhance_model_dir = 'enhance_models'
tissue_model_depths = {'right_ventricle': '16.5M',
                       'adrenal': '25M',
                       'bladder': '12.5M',
                       'cortex': '8.5M',
                       'hippocampus': '8.5M',
                       'lung': '8.5M',
                       'ovary': '5M',
                       'pancreas': '8.5M',
                       'psoas_muscle': '8.5M',
                       'intestine': '8.5M',
                       'spleen': '16.5M',
                       'liver': '50M',
                       'aorta': '25M',
                       'left_ventricle': '25M'}

out_dir = 'universal_enhancer_heatmaps'
os.makedirs(out_dir, exist_ok=True)
n_examples = 10
subplot_inches = 4
anchor_dir = args.anchor_dir
matrix_size = args.matrix_size
step_size = args.step_size

model = keras.models.load_model('models/universal_enhancer_99.h5')
tissue_list = os.listdir(test_dir)

locations = {'alb': {'chr4': 73397114},
             'addl': {'chr2': 70607618},
             'myoz2': {'chr4': 119135832},
             'hsd3b2': {'chr1': 119414931},
             'fabp6': {'chr5': 160187367}}

tissue_models = {}
for tissue in tissue_list:
    depth = tissue_model_depths[tissue]
    print(tissue, depth)
    with open(os.path.join(enhance_model_dir, '%s.json' % depth), 'r') as f:
        enhance = keras.models.model_from_json(f.read())  # load model
    enhance.load_weights(os.path.join(enhance_model_dir, '%s.h5' % depth))  # load model weights
    tissue_models[tissue] = enhance


chr_names = ['chr%d' % i for i in range(1, 23)] + ['chrX', 'chrY']

for gene_name in locations.keys():
    chr_name = list(locations[gene_name].keys())[0]
    loc = locations[gene_name][chr_name]
    anchor_list = pd.read_csv(anchor_dir + '/%s.bed' % chr_name, sep='\t',
                              names=['chr', 'start', 'end', 'anchor'])  # read anchor list file
    start_anchor = anchor_list[(anchor_list['start'] + matrix_size / 2 <= loc) & (loc <= anchor_list['end'] + matrix_size / 2)].index[0] + int(matrix_size / 4)
    end_anchor = start_anchor + matrix_size  #anchor_list[(anchor_list['start'] <= locations[chr_name] + matrix_size / 2) & (locations[chr_name] + matrix_size / 2 <= anchor_list['end'])].index[0]
    print(start_anchor, end_anchor)
    fig, axs = plt.subplots(3, len(tissue_list), figsize=(subplot_inches * len(tissue_list), 3 * subplot_inches))
    for i, tissue_dir in enumerate(tissue_list):
        print(tissue_dir)
        chr_dir = os.path.join(test_dir, tissue_dir)
        chromosome = get_chromosome_file(chr_dir, chr_name)
        # print(chromosome, chr_name)
        sparse_matrix = load_chr_ratio_matrix_from_sparse(chr_dir, chromosome, anchor_dir=anchor_dir,
                                                          chr_name=chr_name, use_raw=False)
        rows = slice(start_anchor, end_anchor)
        tile = sparse_matrix[rows, rows].A

        draw_heatmap(tile, 0, ax=axs[0][i])
        axs[0][i].set_title(tissue_dir)
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])

        tile = np.expand_dims(np.expand_dims(tile, -1), 0)

        enhanced = tissue_models[tissue_dir].predict(tile)[0, ..., 0]
        enhanced = (enhanced + enhanced.T) * 0.5

        draw_heatmap(enhanced, 0, ax=axs[1][i])
        axs[1][i].set_title('enhanced (%s)' % tissue_model_depths[tissue_dir])
        axs[1][i].set_xticks([])
        axs[1][i].set_yticks([])

        pred = model.predict(tile)[0, ..., 0]
        pred = (pred + pred.T) * 0.5

        draw_heatmap(pred, 0, ax=axs[2][i])
        axs[2][i].set_title('universal enhance')
        axs[2][i].set_xticks([])
        axs[2][i].set_yticks([])

    fig.savefig(os.path.join(out_dir, '%s_%s.png' % (gene_name, chr_name)))
    plt.close()
