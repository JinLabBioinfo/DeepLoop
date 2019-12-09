import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

datasets = {'GM_17M':'GM.pp.anchor',
            'GM_33M':'GM.pp.anchor',
            'GM_50M':'GM.pp.anchor',
            'GM_100M':'GM.pp.anchor',
            'GM_200M':'GM.pp.anchor',
            'GM_500M':'GM.pp.anchor'}

colors = {'GM_17M': 'tab:blue',
          'GM_33M': 'tab:orange',
          'GM_50M': 'tab:green',
          'GM_100M': 'tab:red',
          'GM_200M': 'tab:cyan',
          'GM_500M': 'tab:brown'}

title = 'PP'

fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(111)
for dataset in datasets.keys():
    print(dataset)
    before_enhance_filename = 'ROC_' + title + '/' + dataset + '_before_enhance'
    enhanced_filename = 'ROC_' + title + '/' + dataset + '_enhanced'
    before_enhance_tpr = np.load(before_enhance_filename + '_recall.npy')
    before_enhance_fpr = np.load(before_enhance_filename + '_fpr.npy')
    before_enhance_tpr = np.append(before_enhance_tpr, 1)
    before_enhance_fpr = np.append(before_enhance_fpr, 1)
    before_enhance_num_top_loops = np.load(before_enhance_filename + '_top_loops.npy')

    enhanced_tpr = np.load(enhanced_filename + '_recall.npy')
    enhanced_fpr = np.load(enhanced_filename + '_fpr.npy')
    enhanced_tpr = np.append(enhanced_tpr, 1)
    enhanced_fpr = np.append(enhanced_fpr, 1)
    enhanced_num_top_loops = np.load(enhanced_filename + '_top_loops.npy')

    before_enhance_auc = metrics.auc(before_enhance_fpr, before_enhance_tpr)
    after_enhance_auc = metrics.auc(enhanced_fpr, enhanced_tpr)

    print('Before Enhance AUC: %.3f\t After Enhance AUC: %.3f' % (before_enhance_auc, after_enhance_auc))

    color = colors[dataset]

    ax.plot(before_enhance_fpr, before_enhance_tpr, linestyle='--', c=color)
    ax.plot(enhanced_fpr, enhanced_tpr, label=dataset, linestyle='-', c=color)

ax = plt.subplot(111)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linestyle='--', label='tpr=fpr')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('pcHiC ROC %s' % title)
ax.legend(loc='best')

fig.savefig('pcHiC_enhanced_ROC_%s.pdf' % title)
plt.show()
