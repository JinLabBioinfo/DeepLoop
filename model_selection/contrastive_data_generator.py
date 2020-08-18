import os
import sys
import math
import time
import random
import numpy as np
import pandas as pd
from collections import deque
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm import tqdm
from utils import load_chr_ratio_matrix_from_sparse, sorted_nicely, get_chromosome_from_filename


class DataGenerator():
    def __init__(self, data_dir, target_dir, anchor_dir, matrix_size, step_size, limit_2Mb=384, batch_size=16, matrices_per_depth=1, batches_per_chr=128, shuffle=True, diagonal_only=True):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.depth_dirs = os.listdir(self.data_dir)
        self.read_depths = []
        self.read_depth_labels = self.depth_dirs.copy()
        for d in self.depth_dirs:
            if d.endswith('k'):  # thousands of reads
                self.read_depths.append(float(d[:-1]) * 1000)
            elif d.endswith('M'):  # millions of reads
                self.read_depths.append(float(d[:-1]) * 1000000)
            else:
                assert 'Did not recognize read depth from directory name %s' % d
        self.read_depths = np.log(self.read_depths)
        #self.read_depths = self.read_depths / np.max(self.read_depths)   # normalize to 0-1
        #self.read_depths = (np.array(self.read_depths) - np.mean(self.read_depths)) / np.std(self.read_depths)
        self.depth_dirs = [os.path.join(self.data_dir, d) for d in self.depth_dirs]  # make depth path absolute
        self.anchor_dir = anchor_dir
        self.anchor_lists = {}
        self.chr_starts = {}
        self.chr_lengths = {}
        self.genome_length = self.get_genome_length()
        self.matrix_size = matrix_size
        self.step_size = step_size
        self.limit_2Mb = limit_2Mb
        self.input_shape = (self.matrix_size, self.matrix_size, 1)
        self.batch_size = batch_size
        self.matrices_per_depth = matrices_per_depth
        self.batches_per_chr = batches_per_chr
        self.diagonal_only = diagonal_only
        self.chr_names = sorted_nicely(list(self.chr_lengths.keys()))
        print(self.chr_names)
        self.anchors = pd.DataFrame()
        genome_length = 0
        for chr_name in self.chr_names:
            self.chr_starts[chr_name] = genome_length
            genome_length += len(self.anchor_lists[chr_name])
            self.anchors = pd.concat([self.anchors, self.anchor_lists[chr_name]])

        n_genomes = 0
        for d in self.depth_dirs:
            n_genomes += len(os.listdir(d))
        matrices_per_genome = genome_length / self.step_size
        self.steps_per_epoch = int(matrices_per_genome * n_genomes / self.batch_size)
        print('%d batches per epoch' % self.steps_per_epoch)

    @staticmethod
    def get_chromosome_file(dir_name, chr_name):
        a2a_files = os.listdir(dir_name)
        for file in a2a_files:
            if chr_name + '.' in file or file.endswith(chr_name):
                return file

    def get_genome_length(self):
        genome_length = 0
        for anchor_file in os.listdir(self.anchor_dir):
            filepath = os.path.join(self.anchor_dir, anchor_file)
            chr_anchors = pd.read_csv(filepath, sep='\t', names=['chr', 'start', 'end', 'anchor'])
            chr_name = anchor_file[:anchor_file.find('.bed')]
            self.anchor_lists[chr_name] = chr_anchors
            self.chr_lengths[chr_name] = len(chr_anchors)
            genome_length += len(chr_anchors)
        print('Each genome has %d rows' % genome_length)
        return genome_length

    def previous_chromosome(self, chr_name):
        index = self.chr_names.index(chr_name)
        return self.chr_names[index - 1]

    def next_chromosome(self, chr_name):
        index = self.chr_names.index(chr_name)
        return self.chr_names[index + 1]

    def load_anchors(self, anchor_dir, chr_name):
        anchor_list = pd.read_csv(os.path.join(anchor_dir, '%s.bed' % chr_name), sep='\t',
                                  names=['chr', 'start', 'end', 'anchor'])  # read anchor list file
        return anchor_list

    def generate_batches(self):
        current_chromosomes = {}  # stores current chromosome sparse matrices for quicker access
        target_chromosomes = {}  # stores all target matrices since they are accessed every batch
        rep_choices = {}  # stores a queue for each depth dir containing the possible replicates to choose from
        anchor_lists = {}  # anchor DataFrames for each chromosome to avoid re-reading files each time
        chr_choices = {}  # stores a queue for each depth dir containing the remaining chromosomes to choose from
        while True:
            x_batch = []
            y_batch = []
            depth_batch = []
            n_batches = 0
            for chromosome in self.chr_lengths.keys():
                print(chromosome)
                if chromosome in anchor_lists.keys():
                    anchor_list = anchor_lists[chromosome]
                else:
                    anchor_list = self.load_anchors(self.anchor_dir, chromosome)
                    anchor_lists[chromosome] = anchor_list
                for row_i in range(0, self.chr_lengths[chromosome] - self.matrix_size, self.step_size):
                    for col_i in range(row_i, self.chr_lengths[chromosome] - self.matrix_size, self.step_size):
                        if self.diagonal_only and row_i != col_i:  # only extract symmetric submatrices from diagonal
                            continue
                        if abs(row_i - col_i) > self.limit_2Mb:  # max distance from diagonal with actual values
                            continue
                        for depth_dir, read_depth in zip(self.depth_dirs, self.read_depths):
                            if depth_dir in current_chromosomes.keys():
                                sparse_matrix = current_chromosomes[depth_dir]
                            else:
                                if depth_dir not in rep_choices.keys() or len(rep_choices[depth_dir]) == 0:
                                    rep_choices[depth_dir] = deque()
                                    for rep in os.listdir(depth_dir):
                                        if 'summary' not in rep:
                                            rep_choices[depth_dir].append(os.path.join(depth_dir, rep))
                                rep_dir = rep_choices[depth_dir].popleft()

                                chr_file = self.get_chromosome_file(rep_dir, chromosome)
                                try:
                                    sparse_matrix = load_chr_ratio_matrix_from_sparse(rep_dir, chr_file,
                                                                                      anchor_dir=self.anchor_dir,
                                                                                      anchor_list=anchor_list,
                                                                                      chr_name=chromosome)
                                except ValueError:
                                    print(rep_dir, chr_file)
                                    continue
                                current_chromosomes[depth_dir] = sparse_matrix

                            if chromosome not in target_chromosomes.keys():
                                target_file = self.get_chromosome_file(self.target_dir, chromosome)
                                target_matrix = load_chr_ratio_matrix_from_sparse(self.target_dir, target_file,
                                                                                  anchor_dir=self.anchor_dir,
                                                                                  anchor_list=anchor_list,
                                                                                  chr_name=chromosome)
                                target_chromosomes[chromosome] = target_matrix
                            else:
                                target_matrix = target_chromosomes[chromosome]

                            rows = slice(row_i, row_i + self.matrix_size)
                            cols = slice(col_i, col_i + self.matrix_size)
                            tile = sparse_matrix[rows, cols].A
                            if random.random() > 0.5:  # flip a coin
                                tile = tile.T  # to flip the matrix
                            tile = np.expand_dims(tile, -1)

                            target_tile = target_matrix[rows, cols].A
                            target_tile = np.expand_dims(target_tile, -1)

                            x_batch.append(tile)
                            y_batch.append(target_tile)
                            depth_batch.append(read_depth)

                            if len(x_batch) >= self.batch_size:
                                n_batches += 1
                                yield np.array(x_batch), np.array(y_batch)
                                x_batch = []
                                y_batch = []
                                depth_batch = []
                    if n_batches == 0:
                        assert 'The batch size can be at most the number of different depths'
                    x_batch = []
                    y_batch = []
                    depth_batch = []
                current_chromosomes = {}  # reset chromosome dict