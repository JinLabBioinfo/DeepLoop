import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')  # necessary when plotting without $DISPLAY
sys.path.append('../')
import numpy as np
from LoopEnhance.enhance_model import EnhanceModel

np.random.seed(42)  # set random seed for reproducible training

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LoopEnhance model')

    parser.add_argument('--downsample_dir', required=True, help='directory containing low depth replicate interaction files (split into chromosomes)')
    parser.add_argument('--target_dir', required=True, help='directory containing target interaction files (split into chromosomes)')
    parser.add_argument('--anchor_dir', required=True, help='directory containing anchor .bed reference files')
    parser.add_argument('--experiment_name', required=True, help='name to be used for saving model and visualizations')
    parser.add_argument('--epochs', required=False, default=50, type=int, help='number of epochs (full tilings of each replicate)')
    parser.add_argument('--learning_rate', required=False, default=1e-3, type=float, help='gradient step size')
    parser.add_argument('--val_downsample_dir', required=False, help='directory containing validation low depth replicate interaction files')
    parser.add_argument('--val_target_dir', required=False, help='directory containing validation target interaction files')
    parser.add_argument('--matrix_size', required=False, default=128, type=int, help='size of symmetric tiles when splitting up genome')
    parser.add_argument('--step_size', required=False, default=128, type=int, help='step size between unique tiles')
    parser.add_argument('--batch_size', required=False, default=4, type=int, help='number of tiles considered in each learning step')
    parser.add_argument('--start_filters', required=False, default=4, type=int, help='number of filters in first layer')
    parser.add_argument('--depth', required=False, default=4, type=int, help='depth of model (number of conv blocks/number of pooling layers in encoder)')
    parser.add_argument('--filter_size', required=False, default=9, type=int, help='size of filters in encoder')
    parser.add_argument('--model_architecture', required=False, default='unet', type=str, help='unet or hicplus')
    parser.add_argument('--activation', required=False, default='relu', type=str, help='non-linear acitvation function (keras strings accepted)')
    parser.add_argument('--loss', required=False, default='mse', type=str, help='loss function for training (keras strings accepted)')

    args = parser.parse_args()

    downsample_dir = args.downsample_dir
    target_dir = args.target_dir
    anchor_dir = args.anchor_dir
    experiment_name = args.experiment_name
    epochs = args.epochs
    learning_rate = args.learning_rate
    matrix_size = args.matrix_size
    step_size = args.step_size
    batch_size = args.batch_size
    start_filters = args.start_filters
    filter_size = args.filter_size
    activation = args.activation
    loss_fn = args.loss
    depth = args.depth
    model_architecture = args.model_architecture
    val_downsample_dir = args.val_downsample_dir
    val_target_dir = args.val_target_dir
    validate = val_downsample_dir is not None and val_target_dir is not None

    enhance = EnhanceModel(matrix_size=matrix_size,
                           step_size=step_size,
                           batch_size=batch_size,
                           epochs=epochs,
                           steps_per_checkpoint=50,
                           steps_per_model_checkpoint=1,
                           start_filters=start_filters,
                           filter_size=filter_size,
                           depth=depth,
                           activation=activation,
                           normalize=False,
                           model_name=experiment_name,
                           loss_type=loss_fn,
                           model_out='models/enhance_model/',
                           model_architecture=model_architecture,
                           verbose=False)

    enhance.train(downsample_dir=downsample_dir,
                  ground_truth_dir=full_data_dir,
                  anchor_dir=anchor_dir,
                  multi_input=True,
                  validate=validate,
                  val_downsample_dir=val_downsample_dir,
                  val_ground_truth_dir=val_ground_truth_dir,
                  val_anchor_dir=val_anchor_dir,
                  val_multi_input=False,
                  learning_rate=learning_rate,
                  save_imgs=True)
