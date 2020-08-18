import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.layers import Dense, Flatten, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from vis.visualization import visualize_activation, visualize_saliency, visualize_cam, overlay
from vis.utils import utils

from model_selection.downsample_data_generator import DataGenerator
from cnn_architectures import Autoencoder
from utils import load_chr_ratio_matrix_from_sparse, draw_heatmap


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--target_dir', required=True, type=str)
parser.add_argument('--anchor_dir', required=True, type=str)
parser.add_argument('--matrix_size', required=False, type=int, default=128)
parser.add_argument('--step_size', required=False, type=int, default=64)
parser.add_argument('--depth', required=False, type=int, default=4)
parser.add_argument('--filters', required=False, type=int, default=4)
parser.add_argument('--n_epochs', required=False, type=int, default=20)
parser.add_argument('--pretrain', required=False, type=bool, default=False)
parser.add_argument('--base_model', required=False, type=str, default='enhance_CP_GZ_001_with_fake.h5')
args = parser.parse_args()

data_dir = args.data_dir
target_dir = args.target_dir
anchor_dir = args.anchor_dir
matrix_size = args.matrix_size
step_size = args.step_size
depth = args.depth
filters = args.filters
n_epochs = args.n_epochs
pretrain = args.pretrain
base_model = args.base_model

data_generator = DataGenerator(data_dir, target_dir, anchor_dir, matrix_size, step_size)
print('Input shape:', data_generator.input_shape)

architectures = Autoencoder(matrix_size, step_size, start_filters=filters, filter_size=9, depth=depth)  # params from LoopEnhance
enhance = architectures.get_unet_model()
enhance.load_weights(base_model)
baseline_enhance = architectures.get_unet_model()
baseline_enhance.load_weights(base_model)
latent_layer = enhance.get_layer(name='conv2d_10').output
encoded_shape = latent_layer.get_shape().as_list()[1:]
encoder = Model(enhance.input, latent_layer)
enhance.compile(optimizer=Adam(1e-5), loss='mse')

print(encoder.summary())
print(enhance.summary())

z = Input(encoded_shape)
x = Flatten()(z)
x = Dense(256, activation='selu')(x)
#x = Dropout(0.1)(x)
out = Dense(1, activation='selu')(x)

adversary = Model(z, out)
print(adversary.summary())
adversary.compile(optimizer=Adam(1e-3), loss='mse')
adversary.trainable = False  # don't allow the adversary to update weights when training joint model

joint_model_out = adversary(encoder(enhance.input))
joint_model = Model(enhance.input, [enhance.output, joint_model_out])
joint_model.compile(optimizer=Adam(1e-5), loss='mse', loss_weights=[1, -.1])  # set loss weight to negative to maximize loss

viz_interval = 5000

loss_plot = []
adv_loss_plot = []
ae_loss_plot = []

os.makedirs('enhance_training', exist_ok=True)

batch_loss = []
batch_adv_loss = []
batch_ae_loss = []
batch_i = -1
for x, [y, depths] in data_generator.generate_batches():
    batch_i += 1
    try:
        embeddings = encoder.predict(x)
    except ValueError:  # bad batch, no reads :(
        continue
    adv_loss = adversary.train_on_batch(embeddings, depths)
    joint_loss = joint_model.train_on_batch(x, [y, depths])
    batch_loss.append(joint_loss[0])
    batch_adv_loss.append(adv_loss)
    batch_ae_loss.append(joint_loss[1])

    if batch_i % viz_interval == 0:
        loss_plot.append(np.mean(batch_loss))
        adv_loss_plot.append(np.mean(batch_adv_loss))
        ae_loss_plot.append(np.mean(batch_ae_loss))
        print('%d: Loss: %.3f, Adv: %.3f, AE: %.3f' % (batch_i, loss_plot[-1], adv_loss_plot[-1], ae_loss_plot[-1]))
        batch_loss = []
        batch_adv_loss = []
        batch_ae_loss = []
        fig, axs = plt.subplots(5, len(data_generator.depth_dirs) + 1, figsize=(6 * len(data_generator.depth_dirs) + 2, 6 * 5))
        #example_chr = data_generator.current_chr
        #example_loc = random.randint(0, data_generator.chr_lengths[example_chr] - matrix_size)
        example_chr = 'chr6'
        example_loc = 16193
        rows = slice(example_loc, example_loc + matrix_size)
        target_file = data_generator.get_chromosome_file(target_dir, example_chr)
        target_matrix = load_chr_ratio_matrix_from_sparse(target_dir, target_file, anchor_dir)
        target_tile = target_matrix[rows, rows].A
        for plot_i, (depth_dir, read_depth) in enumerate(zip(data_generator.depth_dirs, data_generator.read_depth_labels)):
            rep_dirs = os.listdir(depth_dir)
            for f in rep_dirs:
                if 'summary' in f:
                    rep_dirs.remove(f)
                    break
            genome = os.path.join(depth_dir, rep_dirs[0])
            chr_file = data_generator.get_chromosome_file(genome, example_chr)
            matrix = load_chr_ratio_matrix_from_sparse(genome, chr_file, anchor_dir)
            tile = matrix[rows, rows].A
            draw_heatmap(tile, 0, ax=axs[0][plot_i])
            axs[0][plot_i].set_title(read_depth)
            axs[0][plot_i].set_xticks([])
            axs[0][plot_i].set_yticks([])

            tile = np.expand_dims(tile, -1)
            tile = np.expand_dims(tile, 0)

            predicted_depth = adversary.predict(encoder.predict(tile))[0]
            enhanced = enhance.predict(tile)[0, ..., 0]
            enhanced = (enhanced + enhanced.T) * 0.5

            draw_heatmap(enhanced, 0, ax=axs[1][plot_i])
            axs[1][plot_i].set_title('%.2f (Predicted)' % predicted_depth)
            axs[1][plot_i].set_xticks([])
            axs[1][plot_i].set_yticks([])

            baseline_enhanced = baseline_enhance.predict(tile)[0, ..., 0]
            baseline_enhanced = (baseline_enhanced + baseline_enhanced.T) * 0.5

            draw_heatmap(baseline_enhanced, 0, ax=axs[2][plot_i])
            axs[2][plot_i].set_xticks([])
            axs[2][plot_i].set_yticks([])

        draw_heatmap(target_tile, 0, ax=axs[1][-1])
        axs[1][-1].set_title('Target')
        axs[1][-1].set_xticks([])
        axs[1][-1].set_yticks([])

        axs[0][-1].plot(loss_plot, label='joint')
        axs[0][-1].plot(adv_loss_plot, label='adv')
        axs[0][-1].plot(ae_loss_plot, label='ae')
        axs[0][-1].set_title('Loss')
        axs[0][-1].legend(loc='best')

        filters, biases = enhance.get_layer(name='conv2d_1').get_weights()

        filter_viz = np.hstack((filters[:, :, 0,  0], np.zeros((9, 1)), filters[:, :, 0,  1]))
        filter_viz = np.vstack((filter_viz, np.zeros((1, 9 * 2 + 1)), np.hstack((filters[:, :, 0,  2], np.zeros((9, 1)), filters[:, :, 0,  3]))))

        im = axs[2][-1].imshow(filter_viz, cmap='RdGy_r')
        axs[2][-1].set_title('First layer filters')
        axs[2][-1].set_xticks([])
        axs[2][-1].set_yticks([])
        divider = make_axes_locatable(axs[2][-1])
        cax = divider.append_axes('bottom', size='15%', pad=0.05)
        cbar1 = fig.colorbar(im, cax=cax, orientation='horizontal')

        for i in range(len(data_generator.depth_dirs) + 1):
            layer_idx = utils.find_layer_idx(enhance, 'activation_10')
            os.makedirs('/tmp/', exist_ok=True)
            grads = visualize_cam(enhance, layer_idx, filter_indices=i, seed_input=tile, backprop_modifier=None)
            #grads = (grads + grads.T) * 0.5
            #draw_heatmap(filter_img, 0, ax=axs[3][i])
            axs[3][i].imshow(grads)
            axs[3][i].set_title('Saliency %d' % i)
            axs[3][i].set_xticks([])
            axs[3][i].set_yticks([])

        for i in range(len(data_generator.depth_dirs) + 1):
            layer_idx = utils.find_layer_idx(enhance, 'activation_10')
            filter_img = visualize_activation(enhance, layer_idx, filter_indices=i)[..., 0]
            #draw_heatmap(filter_img, 0, ax=axs[3][i])
            axs[4][i].imshow(filter_img, cmap='Reds')
            axs[4][i].set_title('Encoded %d' % i)
            axs[4][i].set_xticks([])
            axs[4][i].set_yticks([])

        n_matrices = int(data_generator.batch_size * (batch_i + 1))

        fig.suptitle('Trained on %d matrices' % n_matrices)

        fig.savefig('enhance_training/%d.png' % (batch_i))
        plt.close()
        enhance.save('models/universal_enhancer_%d.h5' % batch_i)
        adversary.save('models/adversary_%d.h5' % batch_i)