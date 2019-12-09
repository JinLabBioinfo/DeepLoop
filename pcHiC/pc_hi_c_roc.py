import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import open_full_genome


def anchor_ids_to_int(df, cols):
    for col in cols:
        df[col] = df[col].map(lambda x: int(x.replace('A_', '')))  # convert all anchor id strings to integers

title = str(sys.argv[1])
step_size = int(sys.argv[2])
promoter_anchors_file = sys.argv[3]
anchor_to_anchor_dir = sys.argv[4]
max_distance = None
min_distance = None
if len(sys.argv) > 5:
    min_distance = int(sys.argv[5])
    max_distance = int(sys.argv[6])

os.makedirs('ROC_' + title, exist_ok=True)

if title == 'PO':
    pc_filename = 'GM.po'
else:
    pc_filename = 'GM.pp'

datasets = {'GM_17M': pc_filename,
            'GM_33M': pc_filename,
            'GM_50M': pc_filename,
            'GM_100M': pc_filename,
            'GM_200M': pc_filename,
            'GM_500M': pc_filename}

promoter_anchors = pd.read_csv(promoter_anchors_file, sep='\t', names=['a1'])
anchor_ids_to_int(promoter_anchors, ['a1'])
print('%d promoter anchors' % len(promoter_anchors))
for tissue in os.listdir(anchor_to_anchor_dir):
    print(tissue)
    pc_data = pc_filename
    pc_loops = pd.read_csv(pc_data, sep='\t',
                           names=['chr', 'a1', 'a1_start', 'a1_end', 'a2', 'a2_start', 'a2_end', 'pval'],
                           usecols=['a1', 'a1_start', 'a1_end', 'a2', 'a2_start', 'a2_end', 'pval'])
    anchor_ids_to_int(pc_loops, ['a1', 'a2'])
    print('%d total pcHiC interactions' % (len(pc_loops)))
    pc_loops = pc_loops[(abs(pc_loops['a2_start'] - pc_loops['a1_end']) <= 2e6) & (abs(pc_loops['a2_start'] - pc_loops['a1_end']) > 15000)]  # filter loops further than 15kb and within 2Mb
    if max_distance is not None and min_distance is not None:
        pc_loops = pc_loops[(abs(pc_loops['a2_start'] - pc_loops['a1_end']) <= max_distance) & (abs(pc_loops['a2_start'] - pc_loops['a1_end']) > min_distance)]
    swapped_anchors = pc_loops.copy(deep=True)
    swapped_anchors['a1'], swapped_anchors['a2'] = swapped_anchors['a2'], swapped_anchors['a1']
    pc_loops = pd.concat([pc_loops, swapped_anchors]).drop_duplicates()
    pc_loops = pc_loops[pc_loops['a1'] < pc_loops['a2']]
    pc_loops.drop_duplicates(inplace=True)

    fig = plt.figure(figsize=(15, 5))
    tpr = []
    fpr = []
    tps = []
    precision = []
    num_top_loops = []
    loops = open_full_genome(anchor_to_anchor_dir)
    anchor_ids_to_int(loops, ['a1', 'a2'])
    loops = loops[loops['a1'] < loops['a2']]
    print('%d anchor to anchor interactions in full genome' % len(loops))

    significant = pc_loops[pc_loops['pval'] >= 2][['a1', 'a2']]
    insignificant = pc_loops[pc_loops['pval'] < 2][['a1', 'a2']]
    # some of our anchors overlap with both significant and insignificant pcHiC pairs
    overlap = pd.merge(significant, insignificant, on=['a1', 'a2'])
    insignificant = pd.concat([insignificant, overlap]).drop_duplicates(keep=False)

    pc_loops = pd.concat([significant, insignificant])
    loops = pd.merge(loops, pc_loops, on=['a1', 'a2'], how='inner').dropna().drop_duplicates()[['a1', 'a2', 'ratio']]
    significant = pd.merge(significant, loops, on=['a1', 'a2'], how='inner').dropna().drop_duplicates()
    insignificant = pd.merge(insignificant, loops, on=['a1', 'a2'], how='inner').dropna().drop_duplicates()

    print('%d significant %s interactions' % (len(significant), title))
    print('%d insiginifcant %s interactions' % (len(insignificant), title))
    print('%d total pcHiC interactions' % (len(insignificant) + len(significant)))
    print('%d %s interactions in full genome' % (len(loops), title))

    loops.sort_values(by='ratio', ascending=False, inplace=True)
    total_promoter_loops = len(loops)
    for top_promoters in np.arange(0, total_promoter_loops, step_size):
        promoter_loops = loops[0: top_promoters][['a1', 'a2']]
        non_promoter_loops = loops[top_promoters:][['a1', 'a2']]
        num_negative_loops = total_promoter_loops - top_promoters
        num_top_loops.append(top_promoters)
        true_positives = pd.merge(significant, promoter_loops, on=['a1', 'a2'])  # overlapping positives
        true_negatives = pd.merge(insignificant, non_promoter_loops, on=['a1', 'a2'])  # overlapping negatives
        false_positives = pd.merge(insignificant, promoter_loops,on=['a1', 'a2'])  # we call a loop that is insignificant in pcHiC
        false_negatives = pd.merge(significant, non_promoter_loops, on=['a1', 'a2'])  # we do not call a loop that is significant in pcHiC

        tp = len(true_positives)
        tn = len(true_negatives)
        fp = len(false_positives)
        fn = len(false_negatives)

        tps.append(tp)
        tpr.append(0 if tp + fn == 0 else tp / (tp + fn))
        fpr.append(0 if tn + fp == 0 else fp / (tn + fp))
        precision.append(1 if tp + fp == 0 else tp / (tp + fp))

        print('TP: %d, TN: %d, FP: %d, FN: %d' % (tp, tn, fp, fn))
        print('TPR: %.3f, FPR: %.3f\n' % (tpr[-1], fpr[-1]))
    if 'denoise' not in anchor_to_anchor_dir:
        dataset = 'before_denoise'
    else:
        dataset = 'denoised'

    np.save('ROC_' + title + '/' + dataset + '_tp.npy', np.array(tps))
    np.save('ROC_' + title + '/' + dataset + '_recall.npy', np.array(tpr))
    np.save('ROC_' + title + '/' + dataset + '_precision.npy', np.array(precision))
    np.save('ROC_' + title + '/' + dataset + '_fpr.npy', np.array(fpr))
    np.save('ROC_' + title + '/' + dataset + '_top_loops.npy', np.array(num_top_loops))

