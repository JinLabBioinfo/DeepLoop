import math
import os
import re
import cv2
import random
import pickle
import numpy as np
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import scipy.sparse
import scipy.ndimage

chromosome_labels = {'chr1': 0, 'chr2': 1, 'chr3': 2, 'chr4': 3, 'chr5': 4, 'chr6': 5, 'chr7': 6, 'chr8': 7, 'chr9': 8,
                     'chr10': 9, 'chr11': 10, 'chr12': 11, 'chr13': 12, 'chr14': 13, 'chr15': 14, 'chr16': 15, 'chr17': 16, 'chr18': 17,
                     'chr19': 18, 'chr20': 19, 'chr21': 20, 'chr22': 21, 'chrX': 22, 'chrY': 23}

data_dir = 'data/'
sparse_data_dir = 'data/sparse/'
try:
    os.mkdir(data_dir)
except FileExistsError:
    pass
try:
    os.mkdir(sparse_data_dir)
except FileExistsError:
    pass


def open_anchor_to_anchor(filename):
    '''
    Read a tab delimited anchor to anchor file as a DataFrame
    Args:
        filename (:obj:`str`) : full path to anchor to anchor file

    Returns:
        ``pandas.DataFrame``: if reading a normalized anchor to anchor file, columns are ``a1 a2 obs exp ratio``
        and if reading a denoised or enhanced anchor to anchor file, columns are ``a1 a2 ratio``
    '''
    try:  # if before denoise top loops
        df = pd.read_csv(filename,
                         sep='\t',
                         names=['anchor1', 'anchor2', 'obs', 'exp'])
        df['ratio'] = (df['obs'] + 5) / (df['exp'] + 5)
    except ValueError:  # after denoise has no obs or exp
        df = pd.read_csv(filename,
                         sep='\t',
                         names=['anchor1', 'anchor2', 'ratio'],
                         usecols=['anchor1', 'anchor2', 'ratio'])
    return df

def open_full_genome(data_dir):
    '''

    Args:
        data_dir:

    Returns:

    '''
    genome = pd.DataFrame()
    print('Opening genome-wide anchor to anchor...')
    for chr_file in os.listdir(data_dir):
        if 'anchor_2_anchor' in chr_file or 'denoised.anchor.to.anchor' in chr_file:
            print(chr_file)
            genome = pd.concat([genome, open_anchor_to_anchor(data_dir + '/' + chr_file)])
    return genome


def get_chromosome_from_filename(filename):
    """
    Extract the chromosome string from any of the file name formats we use

    Args:
        filename (:obj:`str`) : name of anchor to anchor file

    Returns:
        Chromosome string of form chr<>
    """
    chr_index = filename.find('chr')  # index of chromosome name
    if chr_index == 0:  # if chromosome name is file prefix
        return filename[:filename.find('.')]
    file_ending_index = filename.rfind('.')  # index of file ending
    if chr_index > file_ending_index:  # if chromosome name is file ending
        return filename[chr_index:]
    else:
        return filename[chr_index: file_ending_index]


def save_samples(input_dir, target_dir, matrix_size, multi_input=False, dir_3=None, combined_dir=None, anchor_dir=None, name='sample', chr_index=0, locus=18000, force_symmetry=True):
    """
    Saves sample matrices for use in training visualizations

    Args:
        input_dir (:obj:`str`) : directory containing input anchor to anchor files
        target_dir (:obj:`str`) : directory containing target anchor to anchor files
        matrix_size (:obj:`int`) : size of each sample matrix
        multi_input (:obj:`bool`) : set to True to save samples from each of the multiple input sets in ``input_dir``
        dir_3 (:obj:`str`) : optional directory containing third set of input anchor to anchor files
        combined_dir (:obj:`str`) : optional directory containing combined target anchor to anchor files
        anchor_dir (:obj:`str`) : directory containing anchor reference ``.bed`` files
        name (:obj:`str`) : each saved sample file will begin with this string
        chr_index (:obj:`int`) : index of chromosome to save samples from
        locus (:obj:`int`) : index of anchor to save samples from
    """
    global data_dir
    global sparse_data_dir
    try:
        os.mkdir(sparse_data_dir)
    except FileExistsError as e:
        pass
    if multi_input:
        input_folder_1 = os.listdir(input_dir)[0] + '/'
        input_folder_2 = os.listdir(input_dir)[1] + '/'
        try:
            input_folder_3 = os.listdir(input_dir)[2] + '/'
        except IndexError:
            pass
        chr_name = get_chromosome_from_filename(os.listdir(input_dir + input_folder_1)[chr_index])
    else:
        chr_name = get_chromosome_from_filename(os.listdir(input_dir)[chr_index])
    print('Saving samples from', chr_name, '...')
    if (name == 'enhance' or name == 'val_enhance') and multi_input:
        matrix_1 = load_chr_ratio_matrix_from_sparse(input_dir + input_folder_1, os.listdir(input_dir + input_folder_1)[chr_index], anchor_dir, force_symmetry=force_symmetry)
        matrix_2 = load_chr_ratio_matrix_from_sparse(target_dir, os.listdir(target_dir)[chr_index], anchor_dir, force_symmetry=force_symmetry)
        matrix_3 = None
        combined_matrix = None
    else:
        if multi_input:
            matrix_1 = load_chr_ratio_matrix_from_sparse(input_dir + input_folder_1, os.listdir(input_dir + input_folder_1)[chr_index], anchor_dir, force_symmetry=force_symmetry)
            matrix_2 = load_chr_ratio_matrix_from_sparse(input_dir + input_folder_2, os.listdir(input_dir + input_folder_2)[chr_index], anchor_dir, force_symmetry=force_symmetry)
            matrix_3 = load_chr_ratio_matrix_from_sparse(input_dir + input_folder_3, os.listdir(input_dir + input_folder_3)[chr_index], anchor_dir, force_symmetry=force_symmetry)
            combined_matrix = load_chr_ratio_matrix_from_sparse(target_dir, os.listdir(target_dir)[chr_index], anchor_dir, force_symmetry=force_symmetry)
        else:
            matrix_1 = load_chr_ratio_matrix_from_sparse(input_dir, os.listdir(input_dir)[chr_index], anchor_dir, force_symmetry=force_symmetry)
            matrix_2 = load_chr_ratio_matrix_from_sparse(target_dir, os.listdir(target_dir)[chr_index], anchor_dir, force_symmetry=force_symmetry)
            if dir_3 is not None:
                matrix_3 = load_chr_ratio_matrix_from_sparse(dir_3, os.listdir(dir_3)[chr_index], anchor_dir, force_symmetry=force_symmetry)
            else:
                matrix_3 = None
            if combined_dir is not None:
                combined_matrix = load_chr_ratio_matrix_from_sparse(combined_dir, os.listdir(combined_dir)[chr_index], anchor_dir, force_symmetry=force_symmetry)
            else:
                combined_matrix = None
    i = locus
    j = i  # symmetric matrix for visualizations
    tile_1 = matrix_1[i:i + matrix_size, j:j + matrix_size].A
    tile_2 = matrix_2[i:i + matrix_size, j:j + matrix_size].A
    tile_1 = np.expand_dims(tile_1, 3)  # add channel dimension
    tile_1 = np.expand_dims(tile_1, 0)  # model expects a list of inputs
    tile_2 = np.expand_dims(tile_2, 3)
    tile_2 = np.expand_dims(tile_2, 0)
    if matrix_3 is not None:
        tile_3 = matrix_3[i:i + matrix_size, j:j + matrix_size].A
        tile_3 = np.expand_dims(tile_3, 3)
        tile_3 = np.expand_dims(tile_3, 0)
        np.save('%s%s_3' % (data_dir, name), tile_3)
    if combined_matrix is not None:
        combined_tile = combined_matrix[i:i + matrix_size, j:j + matrix_size].A
        combined_tile = np.expand_dims(combined_tile, 3)
        combined_tile = np.expand_dims(combined_tile, 0)
        np.save('%s%s_combined' % (data_dir, name), combined_tile)
    np.save('%s%s_1' % (data_dir, name), tile_1)
    np.save('%s%s_2' % (data_dir, name), tile_2)


def load_chr_ratio_matrix_from_sparse(dir_name, file_name, anchor_dir, anchor_list=None, chr_name=None, dummy=5, ignore_sparse=False, force_symmetry=False):
    """
    Loads data as a sparse matrix by either reading a precompiled sparse matrix or an anchor to anchor file which is converted to sparse CSR format.
    Ratio values are computed using the observed (obs) and expected (exp) values:

    .. math::
       ratio = \\frac{obs + dummy}{exp + dummy}

    Args:
        dir_name (:obj:`str`) : directory containing the anchor to anchor or precompiled (.npz) sparse matrix file
        file_name (:obj:`str`) : name of anchor to anchor or precompiled (.npz) sparse matrix file
        anchor_dir (:obj:`str`) : directory containing the reference anchor ``.bed`` files
        dummy (:obj:`int`) : dummy value to used when computing ratio values
        ignore_sparse (:obj:`bool`) : set to True to ignore precompiled sparse matrices even if they exist

    Returns:
        ``scipy.sparse.csr_matrix``: sparse matrix of ratio values
    """
    global data_dir
    global sparse_data_dir
    if chr_name is None:
        chr_name = get_chromosome_from_filename(file_name)
    sparse_rep_dir = dir_name[dir_name[: -1].rfind('/') + 1:]  # directory where the pre-compiled sparse matrices are saved
    try:
        os.mkdir(sparse_data_dir + sparse_rep_dir)
    except FileExistsError:
        pass
    if file_name.endswith('.npz'):  # loading pre-combined and pre-compiled sparse data
        sparse_matrix = scipy.sparse.load_npz(dir_name + file_name)
    else:  # load from file name
        try:
            os.mkdir(sparse_data_dir + sparse_rep_dir)
        except FileExistsError:
            pass
        if file_name + '.npz' in os.listdir(sparse_data_dir + sparse_rep_dir) and not ignore_sparse:  # check if pre-compiled data already exists
            sparse_matrix = scipy.sparse.load_npz(sparse_data_dir + sparse_rep_dir + file_name + '.npz')
        else:  # otherwise generate sparse matrix from anchor2anchor file and save pre-compiled data
            if anchor_list is None:
                if anchor_dir is None:
                    assert 'You must supply either an anchor reference list or the directory containing one'
                anchor_list = pd.read_csv(anchor_dir + '%s.bed' % chr_name, sep='\t',
                                          names=['chr', 'start', 'end', 'anchor'])  # read anchor list file
            matrix_size = len(anchor_list) # matrix size is needed to construct sparse CSR matrix
            anchor_dict = anchor_list_to_dict(anchor_list['anchor'].values)  # convert to anchor --> index dictionary
            try:  # first try reading anchor to anchor file as <a1> <a2> <obs> <exp>
                chr_anchor_file = pd.read_csv(
                    dir_name + file_name,
                    delimiter='\t',
                    names=['anchor1', 'anchor2', 'obs', 'exp'],
                    usecols=['anchor1', 'anchor2', 'obs', 'exp'])  # read chromosome anchor to anchor file
                rows = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_file['anchor1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_file['anchor2'].values)  # convert anchor names to column indices
                ratio = (chr_anchor_file['obs'] + dummy) / (chr_anchor_file['exp'] + dummy)  # compute matrix ratio value
                sparse_matrix = scipy.sparse.csr_matrix((ratio, (rows, cols)), shape=(matrix_size, matrix_size))  # construct sparse CSR matrix
            except:  # otherwise read anchor to anchor file as <a1> <a2> <ratio>
                chr_anchor_file = pd.read_csv(
                    dir_name + file_name,
                    delimiter='\t',
                    names=['anchor1', 'anchor2', 'ratio'],
                    usecols=['anchor1', 'anchor2', 'ratio'])
                rows = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_file['anchor1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_file['anchor2'].values)  # convert anchor names to column indices
                sparse_matrix = scipy.sparse.csr_matrix((chr_anchor_file['ratio'], (rows, cols)), shape=(matrix_size, matrix_size))  # construct sparse CSR matrix
            if force_symmetry:
                sparse_triu = scipy.sparse.triu(sparse_matrix)
                sparse_matrix = sparse_triu + sparse_triu.transpose()
            if not ignore_sparse:
                scipy.sparse.save_npz(sparse_data_dir + sparse_rep_dir + file_name, sparse_matrix)  # save precompiled data
    return sparse_matrix


def split_matrix(input_filename,
                 input_matrix,
                 target_matrix,
                 input_batch,
                 target_batch,
                 matrix_size,
                 step_size,
                 batch_size,
                 n_matrices,
                 start_index,
                 normalize,
                 shuffle,
                 random_steps,
                 diagonal_only,
                 upper_triangular_only):
    """
    Generator function to split input and target sparse matrices into patches which are used for training and prediction.

    Args:
        input_filename (:obj:`str`): name of file which is being used to generate ratio matrix patches
        input_matrix (:obj:`scipy.sparse.csr_matrix`) : sparse CSR input matrix
        target_matrix (:obj:`scipy.sparse.csr_matrix`) : sparse CSR target matrix
        input_batch (:obj:`numpy.array`) : current array of samples in the input batch being generated
        target_batch (:obj:`numpy.array`) : current array of samples in the target batch being generated
        matrix_size (:obj:`int`) : size of each patch
        step_size (:obj:`int`) : size of steps used when generating batches.  Values less than ``matrix size`` will include overlapping regions
        batch_size (:obj:`int`) : number of patches to use in each batch
        n_matrices (:obj:`int`) : current number of matrix patches in the batch being generated
        start_index (:obj:`int`) : starting anchor index of the matrix splitting, ensures batches are not identical across epochs
        normalize (:obj:`bool`) : set to True to normalize all ratio values between ``[0, 1]``
        shuffle (:obj:`bool`) : set to True to randomly split the matrix instead of sliding across sequentially
        random_steps (:obj:`bool`) : set to True add a random offset to each step between patch indices
        diagonal_only (:obj:`bool`) : set to True to only generate patches along the diagonal of the matrix

    Returns:
        (``numpy.array``, ``numpy.array``, ``str``): input batch, target batch, and batch label
    """
    if matrix_size == -1:
        input_matrix = np.expand_dims(np.expand_dims(input_matrix.A, 0), -1)
        target_matrix = np.expand_dims(np.expand_dims(target_matrix.A, 0), -1)
        yield input_matrix, target_matrix, input_filename + '_full_chr'
    else:
        if random_steps:  # random offset from step size intervals
            start_index = np.random.randint(0, step_size)
        row_indices = np.arange(start_index, input_matrix.shape[0], step_size)
        col_indices = np.arange(start_index, input_matrix.shape[1], step_size)
        if shuffle:  # shuffle slicing indices
            np.random.shuffle(row_indices)
            np.random.shuffle(col_indices)
        for i in row_indices:
            for j in col_indices:
                if abs(i - j) > 384:  # max distance from diagonal with actual values
                    continue
                if diagonal_only and i != j:
                    continue
                if upper_triangular_only and i < j:
                    continue
                input_tile = input_matrix[i:i + matrix_size, j:j + matrix_size].A
                target_tile = target_matrix[i:i + matrix_size, j:j + matrix_size].A
                input_tile = np.expand_dims(input_tile, axis=3)
                target_tile = np.expand_dims(target_tile, axis=3)
                input_batch = np.append(input_batch, input_tile)
                target_batch = np.append(target_batch, target_tile)
                n_matrices += 1
                if n_matrices == batch_size:
                    try:
                        input_batch = np.reshape(input_batch, (n_matrices, matrix_size, matrix_size, 1))
                        target_batch = np.reshape(target_batch, (n_matrices, matrix_size, matrix_size, 1))
                        if normalize:
                            input_batch = normalize_matrix(input_batch)
                            target_batch = normalize_matrix(target_batch)

                        yield input_batch, target_batch, input_filename + '_' + str(i)
                    except ValueError:  # reached end of valid values
                        pass
                    finally:
                        input_batch = np.array([])
                        target_batch = np.array([])
                        n_matrices = 0


def generate_batches_from_chr(input_dir,
                              target_dir,
                              matrix_size,
                              batch_size,
                              anchor_dir=None,
                              step_size=64,
                              multi_input=False,
                              shuffle=False,
                              random_steps=False,
                              normalize=False,
                              diagonal_only=False,
                              upper_triangular_only=False,
                              force_symmetry=False,
                              ignore_XY=True,
                              ignore_even_chr=False,
                              ignore_odd_chr=False):
    """
    Generator function which generates batches of input target pairs to train the model:

    .. code-block:: python
       :linenos:

       for epoch_i in range(epochs):
           for input_batch, target_batch, batch_label in generate_batches_from_chr(input_dir,
                                                                                   target_dir,
                                                                                   matrix_size=128,
                                                                                   batch_size=64,
                                                                                   step_size=64,
                                                                                   shuffle=True,
                                                                                   random_steps=True,
                                                                                   anchor_dir=anchor_dir):
                step_start_time = time.time()
                loss = model.train_on_batch(noisy_batch, target_batch)
                print("%d-%d %ds [Loss: %.3f][PSNR: %.3f, Jaccard: %.3f]" %
                          (epoch_i,
                           step_i,
                           time.time() - step_start_time,
                           loss[0],
                           loss[1],
                           loss[2]
                           ))
                step_i += 1

    Args:
        input_dir (:obj:`str`) : directory containing all input data to be generated
        target_dir (:obj:`str`) : directory containing all target data to be generated
        matrix_size (:obj:`int`) : size of each patch that the full ratio matrix is divided into
        batch_size (:obj:`int`) : number of patches to use in each batch
        anchor_dir (:obj:`str`) : directory containing the reference anchor ``.bed`` files
        step_size (:obj:`int`) : size of steps used when generating batches.  Values less than ``matrix size`` will include overlapping regions
        multi_input (:obj:`bool`) : set to True to save samples from each of the multiple input sets in ``input_dir``
        shuffle (:obj:`bool`) : set to True to randomly split the matrix instead of sliding across sequentially
        random_steps (:obj:`bool`) : set to True add a random offset to each step between patch indices
        diagonal_only (:obj:`bool`) : set to True to only generate patches along the diagonal of the matrix
        normalize (:obj:`bool`) : set to True to normalize all ratio values between ``[0, 1]``
        ignore_XY (:obj:`bool`) : set to True to ignore chromosomes X and Y when generating batches
        ignore_even_chr (:obj:`bool`) : set to True to ignore all even numbered chromosomes
        ignore_odd_chr (:obj:`bool`) : set to True to ignore all odd numbered chromosomes

    Returns:
        (``numpy.array``, ``numpy.array``, ``str``): input batch, target batch, and batch label
    """
    input_batch = np.array([])
    target_batch = np.array([])
    if multi_input:
        input_folders = os.listdir(input_dir)  # get list of all folders in input dir
        input_files = sorted(os.listdir(input_dir + input_folders[0]))  # get list of input files (assume all inputs have same name pattern)
        target_files = sorted(os.listdir(target_dir))
    else:
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))

    if shuffle:  # shuffle chromosome file order
        c = list(zip(input_files, target_files))
        random.shuffle(c)
        input_files, target_files = zip(*c)

    if ignore_XY:
        remove_XY = lambda files: [f for f in files if 'chrX' not in f and 'chrY' not in f]
        input_files = remove_XY(input_files)
        target_files = remove_XY(target_files)

    if ignore_odd_chr:
        # fun one-liner to remove all odd-numbered chromosomes
        remove_odds = lambda files: [f for f in files if f[f.index('chr') + 3:f.index('.matrix')].isdigit() and int(f[f.index('chr') + 3:f.index('.matrix')]) % 2 == 0]
        input_files = remove_odds(input_files)
        target_files = remove_odds(target_files)
    elif ignore_even_chr:
        remove_evens = lambda files: [f for f in files if f[f.index('chr') + 3:f.index('.matrix')].isdigit() and int(f[f.index('chr') + 3:f.index('.matrix')]) % 2 != 0]
        input_files = remove_evens(input_files)
        target_files = remove_evens(target_files)

    for input_file, target_file in zip(input_files, target_files):
        n_matrices = 0
        start_index = 0
        if multi_input:
            target_matrix = load_chr_ratio_matrix_from_sparse(target_dir, target_file, anchor_dir, force_symmetry=force_symmetry)
            for input_folder in input_folders:
                input_folder += '/'
                input_matrix = load_chr_ratio_matrix_from_sparse(input_dir + input_folder, input_file, anchor_dir, force_symmetry=force_symmetry)
                for input_batch, target_batch, figure_title in split_matrix(input_filename=input_folder + input_file,
                                                                            input_matrix=input_matrix,
                                                                            target_matrix=target_matrix,
                                                                            input_batch=input_batch,
                                                                            target_batch=target_batch,
                                                                            matrix_size=matrix_size,
                                                                            step_size=step_size,
                                                                            batch_size=batch_size,
                                                                            n_matrices=n_matrices,
                                                                            start_index=start_index,
                                                                            normalize=normalize,
                                                                            shuffle=shuffle,
                                                                            random_steps=random_steps,
                                                                            diagonal_only=diagonal_only,
                                                                            upper_triangular_only=upper_triangular_only):
                    yield input_batch, target_batch, figure_title
        else:
            input_matrix = load_chr_ratio_matrix_from_sparse(input_dir, input_file, anchor_dir, force_symmetry=force_symmetry)
            target_matrix = load_chr_ratio_matrix_from_sparse(target_dir, target_file, anchor_dir, force_symmetry=force_symmetry)
            for input_batch, target_batch, figure_title in split_matrix(input_filename=input_file,
                                                                        input_matrix=input_matrix,
                                                                        target_matrix=target_matrix,
                                                                        input_batch=input_batch,
                                                                        target_batch=target_batch,
                                                                        matrix_size=matrix_size,
                                                                        step_size=step_size,
                                                                        batch_size=batch_size,
                                                                        n_matrices=n_matrices,
                                                                        start_index=start_index,
                                                                        normalize=normalize,
                                                                        shuffle=shuffle,
                                                                        random_steps=random_steps,
                                                                        diagonal_only=diagonal_only,
                                                                        upper_triangular_only=upper_triangular_only):
                yield input_batch, target_batch, figure_title


def get_matrices_from_loci(input_dir,
                           target_dir,
                           matrix_size,
                           loci,
                           anchor_dir=None):
    """
    Generator function for getting sample matrices at specific loci

    Args:
        input_dir (:obj:`str`) : directory containing all input data to be generated
        target_dir (:obj:`str`) : directory containing all target data to be generated
        matrix_size (:obj:`int`) : size of each patch that the full ratio matrix is divided into
        loci (:obj:`dict`) : dictionary of chromosome locus pairs
        anchor_dir (:obj:`str`) : directory containing the reference anchor ``.bed`` files

    Returns:
        (``numpy.array``, ``numpy.array``, ``str``, ``int``, ``int``): input matrix, target matrix, chromosome name, locus, and anchor index
    """
    input_files = sorted_nicely(os.listdir(input_dir))
    target_files = sorted_nicely(os.listdir(target_dir))

    for file_1, file_2 in zip(input_files, target_files):
        chr_name = get_chromosome_from_filename(file_1)
        if chr_name in loci.keys():
            anchor_list = pd.read_csv(anchor_dir + '%s.bed' % chr_name, sep='\t',
                                      names=['chr', 'start', 'end', 'anchor'])  # read anchor list file
        else:
            continue
        input_matrix = load_chr_ratio_matrix_from_sparse(input_dir, file_1, anchor_dir)
        target_matrix = load_chr_ratio_matrix_from_sparse(target_dir, file_2, anchor_dir)

        loci_indices = (anchor_list['start'] <= loci[chr_name]) & (loci[chr_name] <= anchor_list['end']) & (anchor_list['chr'] == chr_name)

        for i, locus in enumerate(loci_indices):
            if locus:
                input_tile = input_matrix[i:i + matrix_size, i:i + matrix_size].A
                target_tile = target_matrix[i:i + matrix_size, i:i + matrix_size].A
                input_tile = np.expand_dims(input_tile, axis=3)
                target_tile = np.expand_dims(target_tile, axis=3)
                input_tile = np.expand_dims(input_tile, axis=0)
                target_tile = np.expand_dims(target_tile, axis=0)

                yield input_tile, target_tile, chr_name, loci[chr_name], i


def get_top_loops(matrix_data_dir, reference_dir, num_top_loops=None, q=None, dummy=5):
    """
    Ranks the ratio values of all chromosomes and computes the cutoff value for taking the top ``num_top_loops`` or the ``q`` th quantile

    Args:
        matrix_data_dir (:obj:`str`) : directory containing the anchor to anchor files used to count loops
        reference_dir (:obj:`str`) : directory containing the reference anchor ``.bed`` files
        num_top_loops (:obj:`str`) : number of top loops to consider
        q (:obj:`str`) : quantile range of loops to consider
        dummy (:obj:`str`) : dummy value to use to calculate each ratio value

    Returns:
        ``float`` : cutoff value for top loops
    """
    global data_dir
    if 'top_loop_values.pickle' in os.listdir(data_dir):
        with open(data_dir + 'top_loop_values.pickle', 'rb') as handle:
            top_loop_values = pickle.load(handle)
    else:
        top_loop_values = {}
    if q is not None:  # select top loops based on quantile not quantity
        if matrix_data_dir + str(q) in top_loop_values.keys():
            genome_min_loop_value = top_loop_values[matrix_data_dir + str(q)]
        else:
            top_loops = np.array([])
            for file in os.listdir(matrix_data_dir):
                sparse = load_chr_ratio_matrix_from_sparse(matrix_data_dir, file, reference_dir, dummy=dummy)
                sparse = scipy.sparse.triu(sparse)
                nonzero_indices = sparse.nonzero()
                top_loops = np.append(top_loops, sparse.tocsr()[nonzero_indices].A)
            genome_min_loop_value = np.quantile(top_loops, q=q)
            top_loop_values[matrix_data_dir + str(q)] = genome_min_loop_value
        print('%s %.4f quantile loops cutoff value: %f' % (matrix_data_dir, q, genome_min_loop_value))
    else:  # select top loops based on rank
        if matrix_data_dir + str(num_top_loops) in top_loop_values.keys():
            genome_min_loop_value = top_loop_values[matrix_data_dir + str(num_top_loops)]
        else:
            top_loops = np.array([])
            for file in os.listdir(matrix_data_dir):
                sparse = load_chr_ratio_matrix_from_sparse(matrix_data_dir, file, reference_dir, dummy=dummy)
                sparse = scipy.sparse.triu(sparse)
                loop_list = np.append(top_loops, sparse.data)
                top_loops = loop_list[np.argsort(-loop_list)[:num_top_loops]]
            genome_min_loop_value = top_loops[-1]
            top_loop_values[matrix_data_dir + str(num_top_loops)] = genome_min_loop_value
        print('%s top %d loops cutoff value: %f' % (matrix_data_dir, num_top_loops, genome_min_loop_value))
    with open(data_dir + 'top_loop_values.pickle', 'wb') as handle:
        pickle.dump(top_loop_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return genome_min_loop_value


def anchor_list_to_dict(anchors):
    """
    Converts the array of anchor names to a dictionary mapping each anchor to its chromosomal index

    Args:
        anchors (:obj:`numpy.array`) : array of anchor name values

    Returns:
        `dict` : dictionary mapping each anchor to its index from the array
    """
    anchor_dict = {}
    for i, anchor in enumerate(anchors):
        anchor_dict[anchor] = i
    return anchor_dict


def anchor_to_locus(anchor_dict):
    """
    Function to convert an anchor name to its genomic locus which can be easily vectorized

    Args:
        anchor_dict (:obj:`dict`) : dictionary mapping each anchor to its chromosomal index

    Returns:
        `function` : function which returns the locus of an anchor name
    """
    def f(anchor):
        return anchor_dict[anchor]
    return f


def sorted_nicely(l):
    """
    Sorts an iterable object according to file system defaults
    Args:
        l (:obj:`iterable`) : iterable object containing items which can be interpreted as text

    Returns:
        `iterable` : sorted iterable
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def normalize_matrix(matrix):
    """
    Normalize ratio values between ``[0, 1]`` using the following function:

    .. math::
       f(x) = 1 - \\frac{1}{1 + x}

    .. image:: _static/normalization_function_plot.PNG
       :scale: 100 %
       :align: center

    Args:
        matrix (:obj:`numpy.array`) : matrix of ratio values

    Returns:
        ``numpy.array`` : matrix of normalized ratio values between ``[0, 1]``
    """
    return 1 - (1 / (1 + matrix))


def denormalize_matrix(matrix):
    """
    Reverse the normalization of a matrix to set all  valid normalized values back to their original ratio values using the following function:

    .. math::

       f^{-1}(x) = \\frac{1}{1 - g(x)} - 1 &\\quad \\mbox{where} &\\quad g(x) = \\begin{cases} 0.98, & \\mbox{if } x > 1 \\\\ 0, & \\mbox{if } x < 0 \\\\ x & \\mbox{ otherwise} \\end{cases}

    We apply the function :math:`g(x)` to remove invalid values that could be in a predicted result and because :math:`f^{-1}(x)` blows up as we approach 1:

    .. image:: _static/denormalization_function_plot.PNG
       :scale: 100 %
       :align: center

    Args:
        matrix (:obj:`numpy.array`) : matrix of normalized ratio values

    Returns:
        ``numpy.array`` : matrix of ratio values
    """
    matrix[matrix > 1] = 0.98
    matrix[matrix < 0] = 0
    return (1 / (1 - matrix)) - 1


def draw_heatmap(matrix, color_scale, ax=None, return_image=False):
    """
    Display ratio heatmap containing only strong signals (values > 1 or 0.98th quantile)

    Args:
        matrix (:obj:`numpy.array`) : ratio matrix to be displayed
        color_scale (:obj:`int`) : max ratio value to be considered strongest by color mapping
        ax (:obj:`matplotlib.axes.Axes`) : axes which will contain the heatmap.  If None, new axes are created
        return_image (:obj:`bool`) : set to True to return the image obtained from drawing the heatmap with the generated color map

    Returns:
        ``numpy.array`` : if ``return_image`` is set to True, return the heatmap as an array
    """
    if color_scale != 0:
        breaks = np.append(np.arange(1.001, color_scale, (color_scale - 1.001) / 18), np.max(matrix))
    elif np.max(matrix) < 2:
        breaks = np.arange(1.001, np.max(matrix), (np.max(matrix) - 1.001) / 19)
    else:
        step = (np.quantile(matrix, q=0.98) - 1) / 18
        up = np.quantile(matrix, q=0.98) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(1.001, up, step), np.max(matrix))

    n_bin = 20  # Discretizes the interpolation into bins
    colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'my_list'
    # Create the colormap
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    norm = matplotlib.colors.BoundaryNorm(breaks, 20)
    # Fewer bins will result in "coarser" colomap interpolation
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(matrix, cmap=cm, norm=norm, interpolation='nearest')
    if return_image:
        plt.close()
        return img.get_array()


def get_heatmap(matrix, color_scale):
    if color_scale != 0:
        breaks = np.append(np.arange(1.001, color_scale, (color_scale - 1.001) / 18), np.max(matrix))
    elif np.max(matrix) < 2:
        breaks = np.arange(1.001, np.max(matrix), (np.max(matrix) - 1.001) / 19)
    else:
        step = (np.quantile(matrix, q=0.98) - 1) / 18
        up = np.quantile(matrix, q=0.98) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(1.001, up, step), np.max(matrix))

    n_bin = 20  # Discretizes the interpolation into bins
    colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'my_list'
    # Create the colormap
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    norm = matplotlib.colors.BoundaryNorm(breaks, 20)
    # Fewer bins will result in "coarser" colomap interpolation
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    heatmap = m.to_rgba(matrix)
    mask = matrix > 1.2
    heatmap[..., -1] = np.ones_like(mask) * mask
    return heatmap


def save_images_to_video(output_name, out_dir):
    """
    Saves all training visualization images to a video file

    Args:
        output_name (:obj:`str`) : filename for the saved video file
    """
    image_folder = 'images'
    video_name = out_dir + output_name + '.avi'

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 29.94, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    last_frame = cv2.imread(os.path.join(image_folder, images[-1]))
    for _ in range(150):
        video.write(last_frame)

    cv2.destroyAllWindows()
    video.release()


def get_model_memory_usage(batch_size, model):
    """
    Estimates the amount of memory required to train the model using the current batch size.

    Args:
        batch_size (:obj:`int`) : number of training samples in each batch
        model (:obj:`keras.models.Model`) : uncompiled Keras model to be trained

    Returns:
        ``float`` : estimated memory usage in GB
    """
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

