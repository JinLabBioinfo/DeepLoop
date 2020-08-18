import os
import gc
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from model_selection.multiple_depths_data_generator import DataGenerator
from utils import load_chr_ratio_matrix_from_sparse


def find_nearest(array):
    array = np.asarray(array)

    def f(x):
        idx = (np.abs(array - x)).argmin()
        return array[idx]
    return f


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--anchor_dir', required=True, type=str)
parser.add_argument('--matrix_size', required=False, type=int, default=512)
parser.add_argument('--step_size', required=False, type=int, default=512)
args = parser.parse_args()

data_dir = args.data_dir
anchor_dir = args.anchor_dir
matrix_size = args.matrix_size
step_size = args.step_size
max_depth = 5000000

data_generator = DataGenerator(data_dir, anchor_dir, matrix_size, step_size, diagonal_only=True, max_depth=max_depth)
test_generator = DataGenerator(data_dir, anchor_dir, matrix_size, step_size, diagonal_only=True, max_depth=max_depth, test=True)

model = keras.models.load_model('model_selector.h5')

for generator in [data_generator, test_generator]:
    classes = list(generator.read_depth_labels)# * data_generator.max_depth))
    read_depths = generator.read_depths
    print(classes)
    class_distributions = {}
    for c in classes:
        class_distributions[c] = []

    true_labels = []
    predictions = []

    for depth_dir, depth in zip(generator.depth_dirs, classes):
        print(depth)
        print(np.exp(generator.read_depths))
        label = classes.index(depth)
        for rep_dir in os.listdir(depth_dir):
            if 'summary' in rep_dir:
                continue
            start_time = time.time()
            votes = []
            weights = []
            abs_rep_dir = os.path.join(depth_dir, rep_dir)
            for chr_file in os.listdir(abs_rep_dir):
                try:
                    matrix = load_chr_ratio_matrix_from_sparse(abs_rep_dir, chr_file, anchor_dir)
                    for i in range(0, matrix.shape[0] - matrix_size, step_size):
                        rows = slice(i, i + matrix_size)
                        tile = matrix[rows, rows].A
                        tile = np.expand_dims(tile, -1)
                        tile = np.expand_dims(tile, 0)
                        pred = model.predict(tile)
                        pred = np.squeeze(np.asarray(pred).ravel())
                        votes.append(pred)
                except ValueError:  # no reads
                    pass
            votes = np.array(votes)
            nearest = np.vectorize(find_nearest(read_depths))(votes)
            #weights = np.array(weights)
            #bagged_votes = np.mean(weights, axis=0)
            #votes = np.argmax(votes, axis=1)
            #print(weights)
            #nearest = np.vectorize(find_nearest(np.array(classes)))(np.rint(votes))
            genome_prediction = stats.mode(nearest, axis=None)[0]
            class_distributions[depth].append(nearest)
            pred_label = np.where(read_depths == genome_prediction)[0]
            #pred_label = read_depths.index(genome_prediction)
            predictions.append(pred_label)
            true_labels.append(label)
            print('%ds: true: %d, pred: %d, %d, %s' % (
            time.time() - start_time, label, pred_label, genome_prediction, abs_rep_dir))

            del matrix  # explicitly remove from memory
            gc.collect()  # for some reason we have a memory leak when loading many sparse matrices

        acc = accuracy_score(true_labels, predictions)
        print('Accuracy:', acc)
        try:
            dist = np.squeeze(np.asarray(class_distributions[depth]).ravel())
            sns.distplot(dist)
            plt.title('%s' % depth)
            plt.xlim(0, None)
            plt.xlabel('predicted downsample depth')
            #plt.xticks(np.arange(0, len(classes)), labels=classes)
            plt.ylabel('density')
            plt.show()
        except np.linalg.LinAlgError:  # singular matrix from only one prediction (this is good!)
            pass
        except ValueError:
            pass

        target_names = ['%s' % d for d in classes]
        #print(classification_report(true_labels, predictions, target_names=target_names))

    acc = accuracy_score(true_labels, predictions)
    print('Accuracy:', acc)

    confusion = confusion_matrix(true_labels, predictions)
    plt.matshow(confusion, cmap='Blues')
    tick_locs = (np.arange(len(classes)) + 0.5) * (len(classes) - 1) / len(classes)
    plt.xticks(ticks=tick_locs, labels=classes)
    plt.yticks(ticks=tick_locs, labels=classes)
    plt.show()

    target_names = ['%s' % d for d in classes]
    print(classification_report(true_labels, predictions, target_names=target_names))

