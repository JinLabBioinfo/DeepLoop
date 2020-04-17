import sys
from LoopDenoise.denoise_model import DenoiseModel
import numpy as np

if __name__ == '__main__':
    noisy_dir = sys.argv[1]
    target_dir = sys.argv[2]
    anchor_dir = sys.argv[3]
    experiment_name = sys.argv[4]
    val_noisy_dir = None
    val_target_dir = None
    if len(sys.argv) > 5:
        val_noisy_dir = sys.argv[5]
        val_target_dir = sys.argv[6]

    np.random.seed(42)  # set random seed for reproducible research

    n2n = DenoiseModel(matrix_size=128,
                       step_size=128,
                       batch_size=4,
                       epochs=50,
                       steps_per_checkpoint=50,
                       steps_per_model_checkpoint=1,
                       start_filters=8,
                       filter_size=13,
                       transpose_filter_size=2,
                       normalize=True,
                       model_name=experiment_name,
                       activation='relu',
                       loss_type='mse',
                       verbose=False)
    n2n.train(noisy_dir=noisy_dir,
              target_dir=target_dir,
              anchor_dir=anchor_dir,
              multi_input=True,
              learning_rate=1e-3,
              val_noisy_dir=val_noisy_dir,
              val_target_dir=val_target_dir,
              val_anchor_dir=anchor_dir,
              val_multi_input=True,
              validate=True,
              save_imgs=True)






