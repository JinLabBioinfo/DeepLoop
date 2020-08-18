import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.layers import Dense, Flatten, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from vis.visualization import visualize_activation, visualize_saliency, visualize_cam
from vis.utils import utils

from model_selection.depth_enhance_data_generator import DataGenerator
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

data_generator = DataGenerator(data_dir, target_dir, anchor_dir, matrix_size, step_size)
print('Input shape:', data_generator.input_shape)

architectures = Autoencoder(matrix_size, step_size, start_filters=filters, filter_size=9, depth=depth)  # params from LoopEnhance
enhance = architectures.get_unet_model()
enhance.load_weights('enhance_CP_GZ_001_with_fake.h5')
baseline_enhance = architectures.get_unet_model()
baseline_enhance.load_weights('enhance_CP_GZ_001_with_fake.h5')
latent_layer = enhance.get_layer(name='conv2d_10').output
encoded_shape = latent_layer.get_shape().as_list()[1:]
encoder = Model(enhance.input, latent_layer)
enhance.compile(optimizer='adam', loss='mse')

print(encoder.summary())
print(enhance.summary())

z = Input(encoded_shape)
x = Flatten()(z)
x = Dense(128, activation='relu')(x)
#x = Dropout(0.1)(x)
out = Dense(1, activation='sigmoid')(x)

adversary = Model(z, out)
print(adversary.summary())
adversary.compile(optimizer=Adam(1e-5), loss='mse')
adversary.trainable = False  # don't allow the adversary to update weights when training joint model

joint_model_out = adversary(encoder(enhance.input))
joint_model = Model(enhance.input, [enhance.output, joint_model_out])
joint_model.compile(optimizer=Adam(1e-5), loss='mse', loss_weights=[1, -1])  # set loss weight to negative to maximize loss

n_reps = 6
viz_interval = 500

loss_plot = []
adv_loss_plot = []
ae_loss_plot = []

os.makedirs('enhance_training', exist_ok=True)

if pretrain:
    pretrain_batches = 1000
    batch_indices = np.arange(0, data_generator.__len__())
    batch_adv_loss = []
    batch_ae_loss = []
    for batch_i in batch_indices:
        x, [y, depths] = data_generator.__getitem__(batch_i)
        try:
            ae_loss = enhance.train_on_batch(x, y)
        except ValueError:
            print('bad batch')
            continue
        print('%d: AE: %.3f' % (batch_i, ae_loss))
        batch_ae_loss.append(ae_loss)
        if batch_i >= pretrain_batches:
            ae_loss_plot.append(np.mean(batch_ae_loss))
            break
    for batch_i in batch_indices:
        x, [y, depths] = data_generator.__getitem__(batch_i)
        try:
            embeddings = encoder.predict(x)
        except ValueError:  # bad batch, no reads :(
            continue
        adv_loss = adversary.train_on_batch(embeddings, depths)
        print('%d: Adv: %.3f' % (batch_i, adv_loss))
        batch_adv_loss.append(adv_loss)
        if batch_i >= pretrain_batches:
            adv_loss_plot.append(np.mean(batch_adv_loss))
            break


for epoch_i in range(n_epochs):
    batch_indices = np.arange(0, data_generator.__len__())
    batch_loss = []
    batch_adv_loss = []
    batch_ae_loss = []
    for batch_i in batch_indices:
        x, [y, depths] = data_generator.__getitem__(batch_i)
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
            print('%d %d: Loss: %.3f, Adv: %.3f, AE: %.3f' % (epoch_i, batch_i, loss_plot[-1], adv_loss_plot[-1], ae_loss_plot[-1]))
            batch_loss = []
            batch_adv_loss = []
            batch_ae_loss = []
            fig, axs = plt.subplots(5, int(len(data_generator.genome_order) / n_reps) + 1, figsize=(6 * n_reps + 2, 6 * 5))
            #example_chr = data_generator.current_chr
            #example_loc = random.randint(0, data_generator.chr_lengths[example_chr] - matrix_size)
            example_chr = 'chr1'
            example_loc = 24036
            for genome_i in range(n_reps):
                genome = data_generator.genome_order[genome_i]
                read_depth = int(genome[-3:])  # depth is stored as zero padded percentage
                label = data_generator.classes.index(read_depth)

                chr_file = data_generator.find_chromosome_file(genome, example_chr)
                matrix = load_chr_ratio_matrix_from_sparse(genome, chr_file, anchor_dir)
                rows = slice(example_loc, example_loc + matrix_size)
                tile = matrix[rows, rows].A
                draw_heatmap(tile, 0, ax=axs[0][genome_i])
                axs[0][genome_i].set_title('%d %%' % read_depth)
                axs[0][genome_i].set_xticks([])
                axs[0][genome_i].set_yticks([])

                tile = np.expand_dims(tile, -1)
                tile = np.expand_dims(tile, 0)

                predicted_depth = int(adversary.predict(encoder.predict(tile))[0] * data_generator.classes[-1])
                enhanced = enhance.predict(tile)[0, ..., 0]
                enhanced = (enhanced + enhanced.T) * 0.5

                draw_heatmap(enhanced, 0, ax=axs[1][genome_i])
                axs[1][genome_i].set_title('%d %% (Predicted)' % predicted_depth)
                axs[1][genome_i].set_xticks([])
                axs[1][genome_i].set_yticks([])

                baseline_enhanced = baseline_enhance.predict(tile)[0, ..., 0]
                baseline_enhanced = (baseline_enhanced + baseline_enhanced.T) * 0.5

                draw_heatmap(baseline_enhanced, 0, ax=axs[2][genome_i])
                axs[2][genome_i].set_xticks([])
                axs[2][genome_i].set_yticks([])
            target_file = data_generator.find_chromosome_file(target_dir, example_chr)
            target_matrix = load_chr_ratio_matrix_from_sparse(target_dir, target_file, anchor_dir)
            target_tile = target_matrix[rows, rows].A
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

            #first_layer = enhance.get_layer(name='conv2d_1')
            '''
            for i in range(architectures.start_filters):
                layer_idx = utils.find_layer_idx(enhance, 'activation_1')
                filter_img = visualize_activation(enhance, layer_idx, filter_indices=i, verbose=True)[..., 0]
                axs[3][i].imshow(filter_img, cmap='Reds')
                axs[3][i + 3].set_title('First Layer %d' % i)
                axs[3][i].set_xticks([])
                axs[3][i].set_yticks([])
            '''

            for i in range(n_reps + 1):
                layer_idx = utils.find_layer_idx(enhance, 'activation_10')
                os.makedirs('/tmp/', exist_ok=True)
                grads = visualize_cam(enhance, layer_idx, filter_indices=i, seed_input=tile, backprop_modifier=None)
                #grads = (grads + grads.T) * 0.5
                #draw_heatmap(filter_img, 0, ax=axs[3][i])
                axs[3][i].imshow(grads)
                axs[3][i].set_title('Saliency %d' % i)
                axs[3][i].set_xticks([])
                axs[3][i].set_yticks([])

            for i in range(n_reps + 1):
                layer_idx = utils.find_layer_idx(enhance, 'activation_10')
                filter_img = visualize_activation(enhance, layer_idx, filter_indices=i)[..., 0]
                #draw_heatmap(filter_img, 0, ax=axs[3][i])
                axs[4][i].imshow(filter_img, cmap='Reds')
                axs[4][i].set_title('Encoded %d' % i)
                axs[4][i].set_xticks([])
                axs[4][i].set_yticks([])

            n_matrices = int(data_generator.batch_size * (batch_i + 1) + data_generator.total_batches * epoch_i)

            fig.suptitle('Trained on %d matrices' % n_matrices)

            fig.savefig('enhance_training/%d_%d.png' % (epoch_i, batch_i))
            plt.close()
    enhance.save('universal_enhancer.h5')
    adversary.save('adversary.h5')