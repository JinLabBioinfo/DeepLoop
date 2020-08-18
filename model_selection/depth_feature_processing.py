import os
import sys
import math
import time
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import load_chr_ratio_matrix_from_sparse, sorted_nicely, get_chromosome_from_filename


class DataProcessor():
    def __init__(self, data_dir, anchor_dir, categorical=False, test=False):
        self.data_dir = data_dir
        self.depth_dirs = os.listdir(self.data_dir)
        print(self.depth_dirs)
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
        #plt.scatter(np.arange(0, len(self.read_depths)), np.sort(self.read_depths))
        #plt.show()
        #self.max_depth = np.max(self.read_depths)
        #self.read_depths = self.read_depths / self.max_depth   # normalize to 0-1
        #self.read_depths = (np.array(self.read_depths) - np.mean(self.read_depths)) / np.std(self.read_depths)
        self.depth_dirs = [os.path.join(self.data_dir, d) for d in self.depth_dirs]  # make depth path absolute
        self.anchor_dir = anchor_dir
        self.anchor_lists = {}
        self.chr_starts = {}
        self.chr_lengths = {}
        self.genome_length = self.get_genome_length()
        self.test = test
        self.chr_names = sorted_nicely(list(self.chr_lengths.keys()))
        print(self.chr_names)
        self.anchors = pd.DataFrame()
        genome_length = 0
        for chr_name in self.chr_names:
            self.chr_starts[chr_name] = genome_length
            genome_length += len(self.anchor_lists[chr_name])
            self.anchors = pd.concat([self.anchors, self.anchor_lists[chr_name]])

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

    def get_data(self, load=True, x_file='sparsities.npy', y_file='depths.npy'):
        if load and x_file in os.listdir('.') and y_file in os.listdir('.'):
            x = np.load(x_file)
            y = np.load(y_file)
        else:
            anchor_lists = {}  # anchor DataFrames for each chromosome to avoid re-reading files each time
            x = []
            y = []

            for depth_dir, read_depth in zip(self.depth_dirs, self.read_depths):
                print(depth_dir, read_depth)
                for rep in os.listdir(depth_dir):
                    if 'summary' not in rep:
                        if 'rep1' not in rep and self.test:
                            continue
                        else:
                            rep_x = []
                            rep_dir = os.path.join(depth_dir, rep)
                            for chromosome in sorted_nicely(self.chr_lengths.keys()):
                                if chromosome in anchor_lists.keys():
                                    anchor_list = anchor_lists[chromosome]
                                else:
                                    anchor_list = self.load_anchors(self.anchor_dir, chromosome)
                                    anchor_lists[chromosome] = anchor_list
                                chr_file = self.get_chromosome_file(rep_dir, chromosome)
                                try:
                                    sparse_matrix = load_chr_ratio_matrix_from_sparse(rep_dir, chr_file,
                                                                                      anchor_dir=self.anchor_dir,
                                                                                      anchor_list=anchor_list,
                                                                                      chr_name=chromosome,
                                                                                      use_raw=False)
                                    rep_x.append(sparse_matrix.max())
                                    s = np.sum(sparse_matrix.data >= 1) / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
                                    rep_x.append(s)
                                except ValueError:
                                    print(rep_dir, chr_file)
                                    rep_x.append(0)
                                    rep_x.append(0)
                                    continue

                            print(rep_x)
                            x.append(rep_x)
                            y.append(read_depth)
            np.save(x_file, np.array(x))
            np.save(y_file, np.array(y))
        return np.array(x), np.array(y)
