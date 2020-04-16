import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import draw_heatmap, anchor_to_locus, anchor_list_to_dict, chromosome_labels, sorted_nicely, get_chromosome_from_filename
import scipy.sparse

parser = argparse.ArgumentParser(description='Combine replicates by p-value')

parser.add_argument('--replicates', required=True)
parser.add_argument('--fulldata', required=True)
parser.add_argument('--anchors', required=True)
parser.add_argument('--output', required=False, default='output/')
parser.add_argument('--results', required=False, default='results/')
parser.add_argument('--pmax', required=False, default=0.05, type=float)
parser.add_argument('--varquantile', required=False, default=0.5, type=float)
parser.add_argument('--dummy', required=False, default=5, type=int)
parser.add_argument('--union', required=False, default=False, type=bool)
parser.add_argument('--heatmaps', required=False, default=False, type=bool)
parser.add_argument('--test', required=False, default=False, type=bool)

argument = parser.parse_args()

rep_dir = argument.replicates  # folder containing replicate anchor to anchor files
full_data_file = argument.fulldata  # full data anchor to anchor file
anchor_dir = argument.anchors  # folder containing anchor reference files
out_dir = argument.output  # folder to save combined anchor to anchor files
results_dir = argument.results  # folder to save histograms and KDEs
p_max = argument.pmax  # max p-value for significant signals
rsd_quantile = argument.varquantile  # max relative standard deviation for reproducibile signals
dummy = argument.dummy  # dummy used in calculating ratio value
union_replicates = argument.union  # keep values significant in only one replicate
plot_heatmaps = argument.heatmaps  # save heatmaps to results folder
test = argument.test  # only load the first few thousand anchors for fast testing

os.makedirs(out_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


def open_anchor_to_anchor(filename):
    try:  # if before denoise top loops
        df = pd.read_csv(filename,
                         sep='\t',
                         names=['anchor1', 'anchor2', 'obs', 'exp', 'p_value'],
                         usecols=['anchor1', 'anchor2', 'obs', 'exp', 'p_value'])
        df['ratio'] = (df['obs'] + dummy) / (df['exp'] + dummy)
    except ValueError:  # after denoise has no obs or exp
        df = pd.read_csv(filename,
                         sep='\t',
                         names=['anchor1', 'anchor2', 'ratio', 'p_value'])
    return df


def open_all_loops(data_dir):
    df = pd.DataFrame()
    for file in sorted_nicely(os.listdir(data_dir)):
        if 'p_val' in file:
            print(file)
            df = pd.concat([df, open_anchor_to_anchor(data_dir + '/' + file)])
    return df


def plot_p_value_heatmaps(anchor_to_anchor_files, combined_matrix, anchor_dict, matrix_size, chr_name, anchor_min, anchor_max):
    indices = np.random.choice(np.arange(0, matrix_size - 128), 5, replace=False)
    for i in indices:
        plot_rows = slice(i, i + 128)
        os.makedirs(results_dir + '%s_%d' % (chr_name, i))
        for plot_i, df in enumerate(anchor_to_anchor_files):
            df['anchor_int1'] = df['anchor1'].map(
                lambda x: int(str(x).replace('A_', '')))
            df['anchor_int2'] = df['anchor2'].map(
                lambda x: int(str(x).replace('A_', '')))
            chr_anchor_to_anchor = df[(df['anchor_int1'] >= anchor_min) & (df['anchor_int1'] <= anchor_max) & (df['anchor_int2'] >= anchor_min) & (df['anchor_int2'] <= anchor_max)]
            print('Converting anchors to indices...')
            rows = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_to_anchor['anchor1'].values)
            cols = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_to_anchor['anchor2'].values)

            print('Converting dataframe to sparse COO matrix...')
            print('P-Values...')
            sparse_matrix = scipy.sparse.csr_matrix((chr_anchor_to_anchor['p_value'], (rows, cols)),
                                                    shape=(matrix_size, matrix_size))
            print('Normalized...')
            ratio_matrix = scipy.sparse.csr_matrix(((chr_anchor_to_anchor['obs'] + dummy) / (chr_anchor_to_anchor['exp'] + dummy), (rows, cols)),
                                                    shape=(matrix_size, matrix_size))
            print('Raw...')
            raw_matrix = scipy.sparse.csr_matrix((chr_anchor_to_anchor['obs'], (rows, cols)),
                                                    shape=(matrix_size, matrix_size))


            p_values = sparse_matrix[plot_rows, plot_rows].A
            ratio_values = ratio_matrix[plot_rows, plot_rows].A
            raw_values = raw_matrix[plot_rows, plot_rows].A

            p_values = np.ones_like(p_values) - p_values

            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticks([])
            draw_heatmap(raw_values, 0, ax=ax)
            plt.axis('off')
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(results_dir + '%s_%d/raw_%d.png' % (chr_name, i, plot_i), bbox_inches=extent)
            plt.close()

            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticks([])
            ax.imshow(p_values, cmap='bwr')
            plt.axis('off')
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(results_dir + '%s_%d/p_val_%d.png' % (chr_name, i, plot_i), bbox_inches=extent)
            plt.close()

            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticks([])
            draw_heatmap(ratio_values, 0, ax=ax)
            plt.axis('off')
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(results_dir + '%s_%d/ratio_%d.png' % (chr_name, i, plot_i), bbox_inches=extent)
            plt.close()

        combined_heatmap = combined_matrix[plot_rows, plot_rows].A
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        draw_heatmap(combined_heatmap, 0, ax=ax)
        plt.axis('off')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(results_dir + '%s_%d/combined.png' % (chr_name, i), bbox_inches=extent)
        plt.close()


def get_loop_data(masks, full_data_mask):
    print('Full data:', full_data_mask.sum())
    for i, mask in enumerate(masks):
        print('Rep %d:' % i, mask.sum())
        print('Rep %d & Full data:', (mask & full_data_mask).sum())



for chr_anchor_file in os.listdir(full_data_file):

    anchor_to_anchor_file_list = []  # store anchor to anchor replicates in list

    for rep_folder in os.listdir(rep_dir):
        print(rep_folder)
        anchor_to_anchor = open_anchor_to_anchor(rep_dir + rep_folder + '/' + chr_anchor_file)
        anchor_to_anchor_file_list.append(anchor_to_anchor)

    combined_anchor_to_anchor = pd.DataFrame()
    combined_anchor_to_anchor['anchor1'] = anchor_to_anchor_file_list[0]['anchor1']
    combined_anchor_to_anchor['anchor2'] = anchor_to_anchor_file_list[0]['anchor2']
    masks = []
    print('Combining values...')

    for i, df in enumerate(anchor_to_anchor_file_list):
        masks.append((df['p_value'] < p_max))  # masks for signals that are significant

    # load full data
    full_anchor_to_anchor = open_anchor_to_anchor(full_data_file + chr_anchor_file)
    full_data_mask = full_anchor_to_anchor['p_value'] < p_max  # mask for significant values in full data

    get_loop_data(masks, full_data_mask)

    if union_replicates:
        significant_mask = ((masks[0] | masks[1] | masks[2]) & full_data_mask)  # significant in one replicate AND full data
        print('Total:', significant_mask.sum())
    else:
        significant_mask = (masks[0] & masks[1]) | (masks[1] & masks[2]) | (masks[0] & masks[2])  # significant in at least two replicates

    ratio_mask = significant_mask & full_data_mask
    print('Percent signals kept: %d%%' % int(ratio_mask.sum() / len(combined_anchor_to_anchor) * 100))
    combined_anchor_to_anchor['ratio'] = 0
    combined_anchor_to_anchor.loc[ratio_mask, 'ratio'] = full_anchor_to_anchor.loc[ratio_mask, 'ratio']

    # create rows of anchor int values to compare to reference .bed files and split into chromosomes
    combined_anchor_to_anchor['anchor_int1'] = combined_anchor_to_anchor['anchor1'].map(lambda x: int(str(x).replace('A_', '')))
    combined_anchor_to_anchor['anchor_int2'] = combined_anchor_to_anchor['anchor2'].map(lambda x: int(str(x).replace('A_', '')))

    chr_index = 0
    chr_file = get_chromosome_from_filename(chr_anchor_file)
    anchor_list = pd.read_csv(anchor_dir + '%s.bed' % chr_file, sep='\t',
                              names=['chr', 'start', 'end', 'anchor'])  # read anchor list file
    anchor_list['anchor_int'] = anchor_list['anchor'].map(
        lambda x: int(str(x).replace('A_', '')))
    anchor_dict = anchor_list_to_dict(anchor_list['anchor'].values)
    matrix_size = len(anchor_list)
    anchor_min = anchor_list['anchor_int'].min()
    anchor_max = anchor_list['anchor_int'].max()
    print('Anchors:', anchor_min, anchor_max)
    chr_anchor_to_anchor = combined_anchor_to_anchor[(combined_anchor_to_anchor['anchor_int1'] >= anchor_min) & (combined_anchor_to_anchor['anchor_int1'] <= anchor_max) & (combined_anchor_to_anchor['anchor_int2'] >= anchor_min) & (combined_anchor_to_anchor['anchor_int2'] <= anchor_max)]
    chr_anchor_to_anchor[['anchor1', 'anchor2', 'ratio']].to_csv(out_dir + 'anchor.to.anchor.p%.2f.%s' % (p_max, chr_file), sep='\t', header=False, index=False)
    chr_index += matrix_size
    if plot_heatmaps:
        print('Converting anchors to indices...')
        rows = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_to_anchor['anchor1'].values)
        cols = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_to_anchor['anchor2'].values)

        print('Converting dataframe to sparse COO matrix...')
        sparse_matrix = scipy.sparse.csr_matrix((chr_anchor_to_anchor['ratio'], (rows, cols)),
                                                shape=(matrix_size, matrix_size))

        plot_p_value_heatmaps(anchor_to_anchor_file_list, sparse_matrix, anchor_dict, matrix_size, chr_file, anchor_min, anchor_max)
