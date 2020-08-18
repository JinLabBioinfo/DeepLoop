import os
import math
import umap
import argparse
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from model_selection.depth_feature_processing import DataProcessor
from utils import sorted_nicely, load_chr_ratio_matrix_from_sparse


def find_nearest(array):
    array = np.asarray(array)

    def f(x):
        idx = (np.abs(array - x)).argmin()
        return array[idx]
    return f


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--anchor_dir', required=True, type=str)
args = parser.parse_args()

data_dir = args.data_dir
anchor_dir = args.anchor_dir

data = DataProcessor(data_dir, anchor_dir)
classes = data.read_depth_labels

x, y = data.get_data()
y_true = []
for depth in y:
    depth = math.exp(depth)
    if depth > 1e6:
        y_true.append('%.2fM' % (depth / 1e6))
    else:
        y_true.append('%dk' % int(depth / 1000))

#reg = SVR().fit(x, y)
reg = tree.DecisionTreeRegressor().fit(x, y)

feature_names = []
for chr_name in data.chr_names:
    for name in ['max', 'sparsity']:
        feature_names.append('%s %s' % (chr_name, name))

fig = plt.figure(figsize=(30, 30))
_ = tree.plot_tree(reg,
                   feature_names=feature_names,
                   class_names=classes,
                   filled=True)
plt.savefig('dtree.png', dpi=400)
plt.show()

#coef = reg.coef_

#print(reg.score(x, y))
#print(coef)

y_fit = reg.predict(x)

print(y)
print(y_fit)

for i in range(x.shape[1]):
    feature = x[:, i]
    plt.scatter(feature, y)
    plt.scatter(feature, y_fit)
    plt.savefig('reg_%d.png' % i)
    plt.close()

nearest = np.vectorize(find_nearest(np.exp(data.read_depths)))(np.rint(np.exp(y_fit)))

y_pred = []
for depth in nearest:
    #depth = math.exp(depth)
    if depth > 1e6:
        y_pred.append('%.2fM' % (depth / 1e6))
    else:
        y_pred.append('%dk' % int(depth / 1000))

print(y_true)
print(y_pred)

encoder = LabelEncoder()
encoder.fit(y_true)
y_true = encoder.transform(y_true)
y_pred = encoder.transform(y_pred)

acc = accuracy_score(y_true, y_pred)
print('Accuracy:', acc)

confusion = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.matshow(confusion, cmap='Blues')
tick_locs = (np.arange(len(classes))) #* (len(classes) - 1) / len(classes)
ax.set_xticks(tick_locs)
ax.set_xticklabels(classes)
ax.set_yticks(tick_locs)
ax.set_yticklabels(classes)
plt.show()

target_names = ['%s' % d for d in classes]
print(classification_report(y_true, y_pred, target_names=target_names))

test_dir = '../../../data/anchor_to_anchor/public_low_depth_tissue_data'

for tissue_dir in os.listdir(test_dir):
    chr_dir = os.path.join(test_dir, tissue_dir)
    x = []
    for chromosome in sorted_nicely(os.listdir(chr_dir)):
        chr_name = chromosome[:chromosome.find('.')]
        #print(chromosome, chr_name)
        sparse_matrix = load_chr_ratio_matrix_from_sparse(chr_dir, chromosome, anchor_dir=anchor_dir, chr_name=chr_name,  use_raw=False)
        x.append(sparse_matrix.max())
        s = np.sum(sparse_matrix.data >= 1) / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
        x.append(s)
    x = np.array(x).reshape(1, -1)
    pred = reg.predict(x)
    pred = round(math.exp(pred))
    print(tissue_dir, pred)



