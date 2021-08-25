import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from keras.models import model_from_json
from scipy.sparse import csr_matrix, triu
try:
    from utils.utils import anchor_to_locus, anchor_list_to_dict
except ModuleNotFoundError:
    from utils import anchor_to_locus, anchor_list_to_dict


def locus_to_anchor(anchor_list):
    def f(locus):
        return anchor_list[locus]

    return f


def predict_tile(args):
    model, shared_denoised, shared_overlap, matrix, window_x, window_y = args
    tile = matrix[window_x, window_y].A  # split matrix into tiles
    if tile.shape == (small_matrix_size, small_matrix_size):
        tile = np.expand_dims(tile, 0)  # add channel dimension
        tile = np.expand_dims(tile, 3)  # add batch dimension
        tmp_denoised = np.ctypeslib.as_array(shared_denoised)
        tmp_overlap = np.ctypeslib.as_array(shared_overlap)
        denoised = model.predict(tile).reshape((small_matrix_size, small_matrix_size))
        denoised[denoised < 0] = 0  # remove any negative values
        tmp_denoised[window_x, window_y] += denoised
        tmp_overlap[window_x, window_y] += 1


def sparse_prediction_from_file(model,
                                matrix,
                                anchor_list,
                                small_matrix_size=128,
                                step_size=64,
                                keep_zeros=True):
    input_matrix_size = len(anchor_list)
    denoised_matrix = np.zeros_like(matrix.A)  # matrix to store denoised values
    overlap_counts = np.zeros_like(matrix.A)  # stores number of overlaps per ratio value

    start_time = time.time()

    for i in range(0, input_matrix_size, step_size):
        for j in range(0, input_matrix_size, step_size):
            if abs(i - j) > 384:  # max distance from diagonal with actual values
                continue
            rows = slice(i, i + small_matrix_size)
            cols = slice(j, j + small_matrix_size)
            if i + small_matrix_size >= input_matrix_size:
                rows = slice(input_matrix_size - small_matrix_size, input_matrix_size)
            if j + small_matrix_size >= input_matrix_size:
                cols = slice(input_matrix_size - small_matrix_size, input_matrix_size)
            tile = matrix[rows, cols].A  # split matrix into tiles
            if tile.shape == (small_matrix_size, small_matrix_size):
                tile = np.expand_dims(tile, 0)  # add channel dimension
                tile = np.expand_dims(tile, 3)  # add batch dimension
                denoised = model.predict(tile).reshape((small_matrix_size, small_matrix_size))
                denoised[denoised < 0] = 0  # remove any negative values
                denoised_matrix[rows, cols] += denoised  # add denoised ratio values to whole matrix
                overlap_counts[rows, cols] += 1  # add to all overlap values within tiled region

    # print('Predicted matrix in %d seconds' % (time.time() - start_time))
    # start_time = time.time()
    denoised_matrix = np.divide(denoised_matrix,
                                overlap_counts,
                                out=np.zeros_like(denoised_matrix),
                                where=overlap_counts != 0)  # average all overlapping areas

    denoised_matrix = (denoised_matrix + denoised_matrix.T) * 0.5  # force symmetry

    np.fill_diagonal(denoised_matrix, 0)  # set all diagonal values to 0

    sparse_denoised_matrix = triu(denoised_matrix, format='coo')

    if not keep_zeros:
        sparse_denoised_matrix.eliminate_zeros()

    # print('Averaging/symmetry, and converting to COO matrix in %d seconds' % (time.time() - start_time))

    return sparse_denoised_matrix


def predict_and_write(model,
                      full_matrix_dir,
                      input_name,
                      out_dir,
                      anchor_dir,
                      chromosome,
                      small_matrix_size,
                      step_size,
                      dummy=5,
                      keep_zeros=True,
                      matrices_per_tile=8):
    start_time = time.time()
    anchor_file = anchor_dir + chromosome + '.bed'
    anchor_list = pd.read_csv(anchor_file, sep='\t', names=['chr', 'start', 'end', 'anchor'])  # read anchor list file
    print('Opened anchor .bed file in %d seconds' % (time.time() - start_time))
    start_time = time.time()
    try:  # first try reading anchor to anchor file as <a1> <a2> <obs> <exp>
        chr_anchor_file = pd.read_csv(
            full_matrix_dir + input_name,
            delimiter='\t',
            names=['anchor1', 'anchor2', 'obs', 'exp'],
            usecols=['anchor1', 'anchor2', 'obs', 'exp'])  # read chromosome anchor to anchor file
        chr_anchor_file['ratio'] = (chr_anchor_file['obs'] + dummy) / (
                    chr_anchor_file['exp'] + dummy)  # compute matrix ratio value
    except:  # otherwise read anchor to anchor file as <a1> <a2> <ratio>
        chr_anchor_file = pd.read_csv(
            full_matrix_dir + input_name,
            delimiter='\t',
            names=['anchor1', 'anchor2', 'ratio'],
            usecols=['anchor1', 'anchor2', 'ratio'])

    print('Opened anchor to anchor file in %d seconds' % (time.time() - start_time))

    denoised_anchor_to_anchor = pd.DataFrame()

    start_time = time.time()

    anchor_step = matrices_per_tile * small_matrix_size

    for i in tqdm(range(0, len(anchor_list), anchor_step)):
        anchors = anchor_list[i: i + anchor_step]
        #print(anchors)
        anchor_dict = anchor_list_to_dict(anchors['anchor'].values)  # convert to anchor --> index dictionary
        chr_tile = chr_anchor_file[
            (chr_anchor_file['anchor1'].isin(anchors['anchor'])) & (chr_anchor_file['anchor2'].isin(anchors['anchor']))]
        #print(chr_tile)
        rows = np.vectorize(anchor_to_locus(anchor_dict))(
            chr_tile['anchor1'].values)  # convert anchor names to row indices
        cols = np.vectorize(anchor_to_locus(anchor_dict))(
            chr_tile['anchor2'].values)  # convert anchor names to column indices
        sparse_matrix = csr_matrix((chr_tile['ratio'], (rows, cols)),
                                   shape=(anchor_step, anchor_step))  # construct sparse CSR matrix

        sparse_denoised_tile = sparse_prediction_from_file(model,
                                                           sparse_matrix,
                                                           anchors,
                                                           small_matrix_size,
                                                           step_size,
                                                           keep_zeros=keep_zeros)
        if len(sparse_denoised_tile.row) > 0:
            anchor_name_list = anchors['anchor'].values.tolist()

            anchor_1_list = np.vectorize(locus_to_anchor(anchor_name_list))(sparse_denoised_tile.row)
            anchor_2_list = np.vectorize(locus_to_anchor(anchor_name_list))(sparse_denoised_tile.col)

            anchor_to_anchor_dict = {'anchor1': anchor_1_list,
                                     'anchor2': anchor_2_list,
                                     'denoised': sparse_denoised_tile.data}

            tile_anchor_to_anchor = pd.DataFrame.from_dict(anchor_to_anchor_dict)
            tile_anchor_to_anchor = tile_anchor_to_anchor.round({'denoised': 4})
            denoised_anchor_to_anchor = pd.concat([denoised_anchor_to_anchor, tile_anchor_to_anchor])

    print('Denoised matrix in %d seconds' % (time.time() - start_time))
    start_time = time.time()

    denoised_anchor_to_anchor.to_csv(out_dir + chromosome + '.denoised.anchor.to.anchor', sep='\t', index=False,
                                     header=False)
    print('Wrote anchor to anchor file in %d seconds' % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_matrix_dir', type=str)
    parser.add_argument('--input_name', type=str)
    parser.add_argument('--h5_file', type=str)
    parser.add_argument('--json_file', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--anchor_dir', type=str)
    parser.add_argument('--chromosome', type=str)
    parser.add_argument('--small_matrix_size', type=int, default=128)
    parser.add_argument('--step_size', type=int, default=128)
    parser.add_argument('--dummy', type=int, default=5)
    parser.add_argument('--keep_zeros', action='store_true')
    args = parser.parse_args()

    full_matrix_dir = args.full_matrix_dir
    input_name = args.input_name
    h5_file = args.h5_file
    if args.json_file is not None:
        json_file = args.json_file
    else:
        json_file = args.h5_file.replace('h5', 'json')
    out_dir = args.out_dir
    anchor_dir = args.anchor_dir
    chromosome = args.chromosome
    small_matrix_size = args.small_matrix_size
    step_size = args.step_size
    dummy = args.dummy
    keep_zeros = args.keep_zeros

    os.makedirs(out_dir, exist_ok=True)

    with open(json_file, 'r') as f:
        model = model_from_json(f.read())  # load model
    model.load_weights(h5_file)  # load model weights
    predict_and_write(model, full_matrix_dir, input_name, out_dir, anchor_dir, chromosome, small_matrix_size, step_size,
                      dummy, keep_zeros)
