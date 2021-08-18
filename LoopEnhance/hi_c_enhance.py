import os
import sys
import shutil
import matplotlib
matplotlib.use('Agg')  # necessary when plotting without $DISPLAY
sys.path.append('../')
import numpy as np
from LoopEnhance.enhance_model import EnhanceModel

shutil.rmtree('data/sparse/')

np.random.seed(42)  # set random seed for reproducible research

if __name__ == '__main__':
    downsample_dir = sys.argv[1]
    full_data_dir = sys.argv[2]
    anchor_dir = sys.argv[3]
    experiment_name = sys.argv[4]

    if len(sys.argv) > 5:
        model_architecture = sys.argv[5]
    else:
        model_architecture = 'unet'

    if len(sys.argv) > 6:
        val_downsample_dir = sys.argv[6]
        val_ground_truth_dir = sys.argv[7]
        val_anchor_dir = sys.argv[8]
        validate=True
    else:
        val_downsample_dir = None
        val_ground_truth_dir = None
        val_anchor_dir = None
        validate=False

    enhance = EnhanceModel(matrix_size=128,
                           step_size=128,
                           batch_size=4,
                           epochs=100,
                           steps_per_checkpoint=50,
                           steps_per_model_checkpoint=1,
                           start_filters=4,
                           filter_size=9,
                           depth=4,
                           activation='relu',
                           normalize=False,
                           model_name=experiment_name,
                           loss_type='mse',
                           model_out='models/enhance_model/',
                           model_architecture=model_architecture,
                           verbose=False)

    enhance.train(downsample_dir=downsample_dir,
                  ground_truth_dir=full_data_dir,
                  anchor_dir=anchor_dir,
                  multi_input=True,
                  validate=False,
                  val_downsample_dir=val_downsample_dir,
                  val_ground_truth_dir=val_ground_truth_dir,
                  val_anchor_dir=val_anchor_dir,
                  val_multi_input=False,
                  learning_rate=1e-5,
                  save_imgs=True)
