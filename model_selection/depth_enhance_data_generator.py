import os
import sys
import math
import time
import random
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from utils import load_chr_ratio_matrix_from_sparse, sorted_nicely


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, target_dir, anchor_dir, matrix_size, step_size, limit_2Mb=384, batch_size=4, shuffle=True, categorical=True):
        self.data_dir = data_dir
        self.depth_dirs = os.listdir(self.data_dir)
        self.read_depths = []
        self.read_depth_labels = np.array(self.depth_dirs.copy())
        for d in self.depth_dirs:
            if d.endswith('k'):  # thousands of reads
                self.read_depths.append(float(d[:-1]) * 1000)
            elif d.endswith('M'):  # millions of reads
                self.read_depths.append(float(d[:-1]) * 1000000)
            else:
                assert 'Did not recognize read depth from directory name %s' % d
        self.read_depths = np.log(self.read_depths)
        sorted_indices = np.argsort(self.read_depths)
        self.read_depths = self.read_depths[sorted_indices]
        self.read_depth_labels = self.read_depth_labels[sorted_indices]
        self.depth_dirs = np.array(self.depth_dirs)[sorted_indices]
        print(self.read_depths)
        plt.scatter(np.arange(0, len(self.read_depths)), np.sort(self.read_depths))
        plt.show()
        if categorical:
            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(self.read_depth_labels)
            encoded_Y = encoder.transform(self.read_depth_labels)
            # convert integers to dummy variables (i.e. one hot encoded)
            self.read_depths = np_utils.to_categorical(encoded_Y)
        print(self.read_depths)
        # self.max_depth = np.max(self.read_depths)
        # self.read_depths = self.read_depths / self.max_depth   # normalize to 0-1
        # self.read_depths = (np.array(self.read_depths) - np.mean(self.read_depths)) / np.std(self.read_depths)
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
        self.matrices_per_genome = 0
        for k in range(limit_2Mb):  # count each diagonal
            self.matrices_per_genome += int((self.genome_length - k * step_size) / self.step_size)
        self.batches_per_genome = int(self.matrices_per_genome / self.batch_size)
        print('%d batches per genome' % self.batches_per_genome)
        self.total_genomes = 0
        for rep in self.rep_dirs:
            self.total_genomes += len(self.depth_dirs[rep])
        print('%d total genomes' % self.total_genomes)
        total_matrices = int(self.total_genomes * self.matrices_per_genome)
        print('%d total matrices' % total_matrices)
        print('%d matrices per genome' % self.matrices_per_genome)
        self.total_batches = int(total_matrices / self.batch_size)
        print('%d batches per epoch' % self.total_batches)
        self.batch_indices = np.arange(0, self.total_batches)
        if shuffle:
            np.random.shuffle(self.batch_indices)

        self.genome_order = []
        for rep in self.rep_dirs:
            for depth_dir in self.depth_dirs[rep]:
                self.genome_order.append(os.path.join(rep, depth_dir))
        assert len(self.genome_order) == self.total_genomes

        self.chr_names = sorted_nicely(list(self.chr_lengths.keys()))
        print(self.chr_names)
        self.anchors = pd.DataFrame()
        genome_length = 0
        for chr_name in self.chr_names:
            self.chr_starts[chr_name] = genome_length
            genome_length += len(self.anchor_lists[chr_name])
            self.anchors = pd.concat([self.anchors, self.anchor_lists[chr_name]])

        self.current_genome = self.genome_order[0]  # genome directory for current batch
        self.current_depth = int(self.current_genome[-3:])  # depth is stored as zero padded percentage
        self.current_chr = self.chr_names[0]  # chromosome for current batch, start at chr1

        self.current_file = self.get_chromosome_file(self.current_chr)
        # sparse matrix for current batch
        self.current_matrix = load_chr_ratio_matrix_from_sparse(dir_name=self.current_genome,
                                                                file_name=self.current_file,
                                                                anchor_dir=self.anchor_dir,
                                                                anchor_list=self.anchor_lists[self.current_chr],
                                                                chr_name=self.current_chr)
        self.target_file = self.find_chromosome_file(self.target_dir, self.current_chr)
        self.target_matrix = load_chr_ratio_matrix_from_sparse(dir_name=self.target_dir,
                                                               file_name=self.target_file,
                                                               anchor_dir=self.anchor_dir,
                                                               anchor_list=self.anchor_lists[self.current_chr],
                                                               chr_name=self.current_chr,
                                                               force_symmetry=True)

        self.epoch = 0

    def get_chromosome_file(self, chr_name):
        a2a_files = os.listdir(self.current_genome)
        for file in a2a_files:
            if file.endswith(chr_name):
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

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.total_batches

    def previous_chromosome(self, chr_name):
        index = self.chr_names.index(chr_name)
        return self.chr_names[index - 1]

    def next_chromosome(self, chr_name):
        index = self.chr_names.index(chr_name)
        return self.chr_names[index + 1]

    @staticmethod
    def find_chromosome_file(dir, chr_name):
        for f in os.listdir(dir):
            if chr_name + '.' in f or f.endswith(chr_name):
                return f

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_index = self.batch_indices[index]
        genome_index = int(batch_index / self.batches_per_genome)  # which data directory to open from
        if genome_index >= len(self.genome_order):
            genome_index -= 1
        self.current_genome = self.genome_order[genome_index]
        self.current_depth = int(self.current_genome[-3:])  # depth is stored as zero padded percentage
        # print(index, batch_index, self.current_genome, self.current_depth)
        row_index = batch_index % self.batches_per_genome / self.batches_per_genome #/ self.limit_2Mb
        row_index = int(row_index * self.genome_length)
        col_offset = batch_index % self.batches_per_genome / self.batches_per_genome * self.limit_2Mb
        k = 0
        while col_offset > self.genome_length:
            col_offset -= self.genome_length - (k * self.step_size)
            k += 1
        col_index = row_index + int(col_offset) * self.step_size
        self.current_chr = self.anchors.iloc[row_index]['chr']
        self.current_file = self.get_chromosome_file(self.current_chr)
        row_index -= self.chr_starts[self.current_chr]  # get relative chromosome index
        col_index -= self.chr_starts[self.current_chr]
        if row_index + self.matrix_size >= self.chr_lengths[self.current_chr] or col_index + self.matrix_size >= self.chr_lengths[self.current_chr]:
            #print(k, col_offset, row_index, col_index, self.chr_lengths[self.current_chr])
            row_index -= self.step_size * self.batch_size
            col_index -= self.step_size * self.batch_size
        #print(k, col_offset, row_index, col_index, self.chr_lengths[self.current_chr])
        self.current_matrix = load_chr_ratio_matrix_from_sparse(dir_name=self.current_genome,
                                                                file_name=self.current_file,
                                                                anchor_dir=self.anchor_dir,
                                                                anchor_list=self.anchor_lists[self.current_chr],
                                                                chr_name=self.current_chr)

        self.target_file = self.find_chromosome_file(self.target_dir, self.current_chr)
        self.target_matrix = load_chr_ratio_matrix_from_sparse(dir_name=self.target_dir,
                                                               file_name=self.target_file,
                                                               anchor_dir=self.anchor_dir,
                                                               anchor_list=self.anchor_lists[self.current_chr],
                                                               chr_name=self.current_chr,
                                                               force_symmetry=True)

        x_batch = []
        y_batch = []
        depth_batch = []
        for i in range(self.batch_size):
            row_start_index = row_index + i * self.step_size
            if row_start_index + self.matrix_size > self.chr_lengths[self.current_chr]:  # if next matrix exceed chromosome
                break  # break and return batch as is
            rows = slice(row_start_index, row_start_index + self.matrix_size)
            col_start_index = col_index + i * self.step_size
            if col_start_index + self.matrix_size > self.chr_lengths[self.current_chr]:  # if next matrix exceed chromosome
                break  # break and return batch as is
            cols = slice(col_start_index, col_start_index + self.matrix_size)
            tile = self.current_matrix[rows, cols].A
            if random.random() > 0.5:  # flip a coin
                tile = tile.T  # to flip the matrix
            #plt.imshow(tile, cmap='Reds')
            #plt.show()
            tile = np.expand_dims(tile, -1)  # add channel dim
            target_tile = self.target_matrix[rows, rows].A
            target_tile = np.expand_dims(target_tile, -1)
            x_batch.append(tile)
            y_batch.append(target_tile)
            depth_batch.append(self.current_depth / self.classes[-1])

        return np.array(x_batch), [np.array(y_batch), np.array(depth_batch)]
