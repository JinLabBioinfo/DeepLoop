import os
import sys
import math
import time
import random
import numpy as np
import pandas as pd
import keras
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm import tqdm
from utils.utils import load_chr_ratio_matrix_from_sparse, sorted_nicely


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, anchor_dir, matrix_size, step_size, batch_size=64):
        self.data_dir = data_dir
        self.rep_dirs = os.listdir(self.data_dir)
        self.rep_dirs = [os.path.join(self.data_dir, d) for d in self.rep_dirs]  # make rep path absolute
        self.depth_dirs = {}  # dict storing anchor to anchor directories for each rep at each depth
        self.genomes_per_rep = {}  # dict storing the number of genomes at different depths for each rep
        for rep in self.rep_dirs:
            self.depth_dirs[rep] = os.listdir(rep)  # fill with different depths for each rep
            self.genomes_per_rep[rep] = len(self.depth_dirs[rep])
        self.anchor_dir = anchor_dir
        self.anchor_lists = {}
        self.chr_lengths = {}
        self.genome_length = self.get_genome_length()
        self.matrix_size = matrix_size
        self.step_size = step_size
        self.input_shape = (self.matrix_size, self.matrix_size, 1)
        self.batch_size = batch_size
        self.batches_per_genome = int(self.genome_length / self.batch_size)

        self.genome_order = []
        for rep in self.rep_dirs:
            for depth_dir in self.depth_dirs[rep]:
                self.genome_order.append(os.path.join(rep, depth_dir))

        self.chr_names = sorted_nicely(list(self.chr_lengths.keys()))

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
            genome_length += len(chr_anchors)
            chr_name = anchor_file[:anchor_file.find('.bed')]
            self.anchor_lists[chr_name] = chr_anchors
            self.chr_lengths[chr_name] = len(chr_anchors)
        return genome_length

    def __len__(self):
        """Denotes the number of batches per epoch"""
        total_genomes = 0
        for rep in self.rep_dirs:
            total_genomes += len(self.depth_dirs[rep])
        total_matrices = int(total_genomes * self.genome_length / self.step_size)
        total_batches = int(total_matrices / self.step_size)
        print('%d batches per epoch' % total_batches)
        return total_batches

    def previous_chromosome(self, chr_name):
        index = self.chr_names.index(chr_name)
        return self.chr_names[index - 1]

    def next_chromosome(self, chr_name):
        index = self.chr_names.index(chr_name)
        return self.chr_names[index + 1]

    def __getitem__(self, index):
        """Generate one batch of data"""
        genome_index = int(index / self.batches_per_genome)  # which data directory to open from
        self.current_genome = self.genome_order[genome_index]
        self.current_depth = int(self.current_genome[-3:])  # depth is stored as zero padded percentage
        row_index = index % self.batches_per_genome
        if self.current_chr != 'chr1':  # if anywhere but the beginning of the genome
            row_index -= self.chr_lengths[self.previous_chromosome(self.current_chr)]  # get relative chromosome index
        if row_index > self.chr_lengths[self.current_chr] - self.matrix_size:  # proceed to next chromosome
            self.current_chr = self.next_chromosome(self.current_chr)
            self.current_file = self.get_chromosome_file(self.current_chr)
            self.current_matrix = load_chr_ratio_matrix_from_sparse(dir_name=self.current_genome,
                                                                    file_name=self.current_file,
                                                                    anchor_dir=self.anchor_dir,
                                                                    anchor_list=self.anchor_lists[self.current_chr],
                                                                    chr_name=self.current_chr)
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            start_index = row_index + i * self.step_size
            rows = slice(start_index, start_index + self.matrix_size)
            tile = self.current_matrix[rows, rows].A
            tile = np.expand_dims(tile, -1)  # add channel dim
            x_batch.append(tile)
            y_batch.append(self.current_depth)

        return np.array(x_batch), np.array(y_batch)


