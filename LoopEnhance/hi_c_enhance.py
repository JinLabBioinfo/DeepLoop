import sys
import numpy as np
from enhance_model import EnhanceModel

np.random.seed(42)  # set random seed for reproducible research

if __name__ == '__main__':
    downsample_dir = sys.argv[1]
    full_data_dir = sys.argv[2]
    anchor_dir = sys.argv[3]
    experiment_name = sys.argv[4]

    if len(sys.argv) > 5:
        val_downsample_dir = sys.argv[5]
        val_ground_truth_dir = sys.argv[6]
        val_anchor_dir = sys.argv[7]

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
                           verbose=False)

    enhance.train(downsample_dir=downsample_dir,
                  ground_truth_dir=full_data_dir,
                  anchor_dir=anchor_dir,
                  multi_input=True,
                  validate=True,
                  val_downsample_dir=val_downsample_dir,
                  val_ground_truth_dir=val_ground_truth_dir,
                  val_anchor_dir=val_anchor_dir,
                  val_multi_input=False,
                  learning_rate=1e-5,
                  save_imgs=True)


