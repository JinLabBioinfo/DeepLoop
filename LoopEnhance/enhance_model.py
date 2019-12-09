import os
import glob
import datetime
import time
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cnn_architectures import Autoencoder
from utils import generate_batches_from_chr, save_samples, draw_heatmap, get_model_memory_usage
from utils import save_images_to_video
from utils import normalize_matrix, denormalize_matrix


class EnhanceModel(Autoencoder):
    def __init__(self,
                 matrix_size=300,
                 step_size=64,
                 batch_size=4,
                 epochs=1,
                 load_pretrained_epoch=0,
                 steps_per_checkpoint=10,
                 steps_per_model_checkpoint=1,
                 start_filters=16,
                 filter_size=3,
                 branching_rate=2.,
                 depth=2,
                 dropout=0.5,
                 model_name='enhance',
                 activation='relu',
                 normalize=False,
                 loss_type='mse',
                 model_out='models/enhance/',
                 denoise_model=None,
                 verbose=True):
        super().__init__(matrix_size=matrix_size,
                         step_size=step_size,
                         start_filters=start_filters,
                         filter_size=filter_size,
                         branching_rate=branching_rate,
                         depth=depth,
                         dropout=dropout,
                         normalize=normalize,
                         activation=activation)
        self.batch_size = batch_size
        self.epochs = epochs
        self.load_pretrained_epoch = load_pretrained_epoch
        self.steps_per_checkpoint = steps_per_checkpoint
        self.steps_per_model_checkpoint = steps_per_model_checkpoint
        self.model_name = model_name
        self.loss_type = loss_type
        self.model_out = model_out
        self.denoise_model = denoise_model
        self.verbose = verbose
        self.loops_plot = np.array([])
        self.percent_loops_plot = np.array([])
        if load_pretrained_epoch > 0:
            model_params = self.model_params_to_name()
            with open(self.model_out + self.model_name + model_params + '.json', 'r') as f:
                self.enhance_model = model_from_json(f.read())
            # Load weights into the new model
            self.enhance_model.load_weights(self.model_out + self.model_name + model_params + '_%d_epochs.h5' % load_pretrained_epoch)
        else:
            self.enhance_model = self.get_unet_model()
            print(self.enhance_model.summary())

    def train(self,
              downsample_dir,
              ground_truth_dir,
              anchor_dir,
              multi_input=False,
              val_downsample_dir=None,
              val_ground_truth_dir=None,
              val_anchor_dir=None,
              val_multi_input=False,
              validate=False,
              validation_curves=False,
              learning_rate=1e-4,
              save_imgs=False):
        optimizer = Adam(lr=learning_rate)
        metrics = [self.psnr, self.recovered_loops, self.total_loops_metric]
        self.enhance_model.compile(optimizer=optimizer,
                                   loss=self.loss_type,
                                   metrics=metrics)
        print('Memory Estimates:')
        print(get_model_memory_usage(self.batch_size, self.enhance_model), 'GB\t')
        try:
            files = glob.glob('images/*')
            for f in files:
                os.remove(f)
        except Exception as e:
            print(e)
        try:
            files = glob.glob('results/training_curves/*')
            for f in files:
                os.remove(f)
        except Exception as e:
            print(e)
        if save_imgs:
            save_samples(downsample_dir, ground_truth_dir, matrix_size=self.matrix_size, name='enhance', anchor_dir=anchor_dir, multi_input=multi_input)
            downsample_sample = np.load('data/enhance_1.npy')
            ground_truth_sample = np.load('data/enhance_2.npy')
            if validate:
                save_samples(val_downsample_dir, val_ground_truth_dir, matrix_size=self.matrix_size, name='val_enhance', anchor_dir=anchor_dir, multi_input=val_multi_input)
                val_downsample_sample = np.load('data/val_enhance_1.npy')
                val_ground_truth_sample = np.load('data/val_enhance_2.npy')
            if self.denoise_model is not None:
                if self.normalize:
                    ground_truth_sample = normalize_matrix(ground_truth_sample)
                ground_truth_sample = self.denoise_model.predict(ground_truth_sample)
                if self.normalize:
                    ground_truth_sample = denormalize_matrix(ground_truth_sample)
            try:
                os.mkdir('images/')
            except FileExistsError as e:
                pass
        step_i = 0
        plot_i = 0
        start_time = time.time()
        if self.load_pretrained_epoch > 0:
            self.loss_plot = np.load('results/saved_curves/loss.npy')[:self.load_pretrained_epoch]
            self.psnr_plot = np.load('results/saved_curves/psnr.npy')[:self.load_pretrained_epoch]
            self.loops_plot = np.load('results/saved_curves/loops.npy')[:self.load_pretrained_epoch]
            self.percent_loops_plot = np.load('results/saved_curves/percent_loops.npy')[:self.load_pretrained_epoch]
            if validate:
                self.val_loss_plot = np.load('results/saved_curves/val_loss.npy')[:self.load_pretrained_epoch]
                self.val_psnr_plot = np.load('results/saved_curves/val_psnr.npy')[:self.load_pretrained_epoch]
                self.val_loops_plot = np.load('results/saved_curves/val_loops.npy')[:self.load_pretrained_epoch]
                self.val_percent_loops_plot = np.load('results/saved_curves/val_percent_loops.npy')[:self.load_pretrained_epoch]
        for epoch_i in range(self.epochs):
            batch_loss = np.array([])
            batch_loops = np.array([])
            batch_psnr = np.array([])
            batch_percent_loops = np.array([])
            for downsample_batch, ground_truth_batch, _ in generate_batches_from_chr(downsample_dir,
                                                                                     ground_truth_dir,
                                                                                     self.matrix_size,
                                                                                     self.batch_size,
                                                                                     multi_input=multi_input,
                                                                                     shuffle=True,
                                                                                     step_size=self.step_size,
                                                                                     random_steps=True,
                                                                                     normalize=self.normalize,
                                                                                     anchor_dir=anchor_dir):
                step_start_time = time.time()
                if self.denoise_model is not None:
                    denoise_ground_truth = self.denoise_model.predict(ground_truth_batch)
                    denoise_ground_truth = (denoise_ground_truth + denoise_ground_truth.T) * 0.5
                    loss = self.enhance_model.train_on_batch(downsample_batch, denoise_ground_truth)
                else:
                    loss = self.enhance_model.train_on_batch(downsample_batch, ground_truth_batch)

                if self.verbose:
                    print("%d-%d %ds [Loss: %.3f][PSNR: %.3f, Percent Recovered Loops: %.3f]" %
                          (epoch_i,
                           step_i,
                           time.time() - step_start_time,
                           loss[0],
                           loss[1],
                           loss[2],
                           ))
                batch_loss = np.append(batch_loss, loss[0])
                batch_psnr = np.append(batch_psnr, loss[1])
                batch_percent_loops = np.append(batch_percent_loops, loss[2])
                batch_loops = np.append(batch_loops, loss[3])
                step_i += 1
            self.loss_plot = np.append(self.loss_plot, batch_loss.mean())
            self.psnr_plot = np.append(self.psnr_plot, batch_psnr.mean())
            self.loops_plot = np.append(self.loops_plot, batch_loops.mean())
            batch_percent_loops = batch_percent_loops[~np.isnan(batch_percent_loops)]  # remove any NaN percent values
            self.percent_loops_plot = np.append(self.percent_loops_plot, batch_percent_loops.mean())
            if validation_curves:
                val_loss = np.array([])
                val_psnr = np.array([])
                val_percent_loops = np.array([])
                val_total_loops = np.array([])
                for val_downsample_batch, val_ground_truth_batch, _ in generate_batches_from_chr(val_downsample_dir,
                                                                                                 val_ground_truth_dir,
                                                                                                 self.matrix_size,
                                                                                                 self.batch_size,
                                                                                                 multi_input=val_multi_input,
                                                                                                 shuffle=True,
                                                                                                 step_size=self.step_size,
                                                                                                 random_steps=True,
                                                                                                 normalize=self.normalize,
                                                                                                 anchor_dir=val_anchor_dir):
                    val_metrics = self.enhance_model.test_on_batch(val_downsample_batch, val_ground_truth_batch)
                    val_loss = np.append(val_loss, val_metrics[0])
                    val_psnr = np.append(val_psnr, val_metrics[1])
                    val_percent_loops = np.append(val_percent_loops, val_metrics[2])
                    val_total_loops = np.append(val_total_loops, val_metrics[3])
                self.val_loss_plot = np.append(self.val_loss_plot, np.mean(val_loss))
                self.val_psnr_plot = np.append(self.val_psnr_plot, np.mean(val_psnr))
                val_percent_loops = val_percent_loops[~np.isnan(val_percent_loops)]  # remove any NaN percent values
                self.val_percent_loops_plot = np.append(self.val_percent_loops_plot, np.mean(val_percent_loops))
                self.val_loops_plot = np.append(self.val_loops_plot, np.mean(val_total_loops))
            self.plot_training_visualization(start_time=start_time,
                                             epoch=epoch_i,
                                             step=step_i,
                                             downsample=downsample_sample,
                                             ground_truth=ground_truth_sample,
                                             val_downsample=val_downsample_sample,
                                             val_ground_truth=val_ground_truth_sample,
                                             validate=validate)

            if epoch_i % self.steps_per_model_checkpoint == 0:
                print('%d - Loss %.4f, PSNR: %.2f, Loops: %.2f, Percent Loops: %.4f' % (epoch_i,
                                                                                        self.loss_plot[-1],
                                                                                        self.psnr_plot[-1],
                                                                                        self.loops_plot[-1],
                                                                                        self.percent_loops_plot[-1]))
                if validation_curves:
                    print('\tTest Loss: %.4f, Test PSNR: %.2f, Test Percent Loops: %.4f' % (self.val_loss_plot[-1],
                                                                                            self.val_psnr_plot[-1],
                                                                                            self.val_percent_loops_plot[-1]))
                self.save_epoch_curves(epoch_i, validate=validate)
                self.save_model(epoch_i)
        # Save the models
        self.save_model(epoch_i)
        self.save_results(save_imgs=save_imgs)

    def save_results(self, save_imgs=False):
        out_dir = 'results/enhance/model_experiments/' + self.model_name + '/'
        os.makedirs(out_dir, exist_ok=True)
        if save_imgs:
            save_images_to_video(self.model_name, out_dir)
        np.save(out_dir + 'loss.npy', self.loss_plot)
        np.save(out_dir + 'psnr.npy', self.psnr_plot)
        np.save(out_dir + 'loops.npy', self.loops_plot)
        np.save(out_dir + 'percent_loops.npy', self.percent_loops_plot)
        self.enhance_model.save(out_dir + 'model_checkpoint_%d_epochs.h5' % self.epochs)
        self.enhance_model.save_weights(out_dir + self.model_name + '.h5')
        with open(out_dir + self.model_name + '.json', 'w') as f:
            f.write(self.enhance_model.to_json())

    def save_model(self, epoch):
        try:
            os.mkdir(self.model_out)
        except FileExistsError:
            pass
        model_params = self.model_params_to_name()
        self.enhance_model.save_weights(self.model_out + self.model_name + model_params + '_%d_epochs.h5' % epoch)
        with open(self.model_out + self.model_name + model_params + '.json', 'w') as f:
            f.write(self.enhance_model.to_json())

    def save_epoch_curves(self, epoch_i, validate=False):
        try:
            os.mkdir('results')
        except FileExistsError:
            pass
        try:
            os.mkdir('results/training_curves')
        except FileExistsError:
            pass
        try:
            os.mkdir('results/saved_curves')
        except FileExistsError:
            pass

        np.save('results/saved_curves/loss.npy', self.loss_plot)
        np.save('results/saved_curves/psnr.npy', self.psnr_plot)
        np.save('results/saved_curves/loops.npy', self.loops_plot)
        np.save('results/saved_curves/percent_loops.npy', self.percent_loops_plot)
        if validate:
            np.save('results/saved_curves/val_loss.npy', self.val_loss_plot)
            np.save('results/saved_curves/val_psnr.npy', self.val_psnr_plot)
            np.save('results/saved_curves/val_loops.npy', self.val_loops_plot)
            np.save('results/saved_curves/val_percent_loops.npy', self.val_percent_loops_plot)

        fig = plt.figure(figsize=(24, 8))

        ax = plt.subplot(131)
        ax.plot(self.loss_plot, label='train')
        if validate:
            ax.plot(self.val_loss_plot, label='train')
        ax.set_title('Loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel(self.loss_type)
        ax.legend(loc='best')

        ax = plt.subplot(132)
        ax.plot(self.psnr_plot, label='train')
        if validate:
            ax.plot(self.val_psnr_plot, label='train')
        ax.set_title('PSNR')
        ax.set_xlabel('epoch')
        ax.set_ylabel('dB')
        ax.legend(loc='best')

        ax = plt.subplot(133)
        ax.plot(self.percent_loops_plot, label='train')
        if validate:
            ax.plot(self.val_percent_loops_plot, label='train')
        ax.set_title('Percent recovered loops')
        ax.set_xlabel('epoch')
        ax.set_ylabel('% loops')
        ax.legend(loc='best')

        fig.savefig('results/training_curves/epoch_%d.png' % epoch_i)
        plt.close()

    def plot_training_visualization(self,
                                    start_time,
                                    epoch,
                                    step,
                                    downsample,
                                    ground_truth,
                                    val_downsample,
                                    val_ground_truth,
                                    validate=False):
        public_low_depth = np.load('data/public_1.npy')
        public_high_depth = np.load('data/public_2.npy')
        if self.normalize:
            downsample = normalize_matrix(downsample)
            val_downsample = normalize_matrix(val_downsample)
        enhanced = self.enhance_model.predict(downsample)
        enhanced = (enhanced + enhanced.T) * 0.5
        val_enhanced = self.enhance_model.predict(val_downsample)
        val_enhanced = (val_enhanced + val_enhanced.T) * 0.5
        public_enhanced = self.enhance_model.predict(public_low_depth)
        public_enhanced = (public_enhanced + public_enhanced.T) * 0.5
        if self.normalize:
            enhanced = denormalize_matrix(enhanced)
            val_enhanced = denormalize_matrix(val_enhanced)
            downsample = denormalize_matrix(downsample)
            val_downsample = denormalize_matrix(val_downsample)

        fig = plt.figure()
        fig.suptitle(self.model_name + '\nTrained on ' + str(step * self.batch_size * 2 * self.steps_per_checkpoint) + ' matrices\n' + str(
            datetime.timedelta(seconds=(int(time.time() - start_time)))), size=12)

        ax = plt.subplot(431)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Downsampled', size=6)
        draw_heatmap(downsample[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(432)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Ground Truth', size=6)
        draw_heatmap(ground_truth[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(433)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Enhanced', size=6)
        draw_heatmap(enhanced[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(434)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Test Downsampled', size=6)
        draw_heatmap(val_downsample[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(435)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Test Ground Truth', size=6)
        draw_heatmap(val_ground_truth[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(436)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Test Enhanced', size=6)
        draw_heatmap(val_enhanced[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(437)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Low Depth Cortex', size=6)
        draw_heatmap(public_low_depth[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(438)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('High Depth Reference', size=6)
        draw_heatmap(public_high_depth[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(439)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Enhanced', size=6)
        draw_heatmap(public_enhanced[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(4, 3, 10)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.set_ylabel(self.loss_type, fontsize=6)
        plt.title('Model Objective', size=6)
        ax.plot(self.loss_plot, label='train', linewidth=0.2)
        if validate:
            ax.plot(self.val_loss_plot, label='test', linewidth=0.2)
        ax.legend(loc='best', prop={'size': 4})

        ax = plt.subplot(4, 3, 11)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.set_ylabel('dB', fontsize=6)
        plt.title('PSNR', size=6)
        ax.plot(self.psnr_plot, label='train', linewidth=0.2)
        if validate:
            ax.plot(self.val_psnr_plot, label='test', linewidth=0.2)
        ax.legend(loc='best', prop={'size': 4})

        ax = plt.subplot(4, 3, 12)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.set_ylabel('recovered/total loops', fontsize=6)
        plt.title('Percent Recovered Loops', size=6)
        ax.plot(self.percent_loops_plot, label='train', linewidth=0.2)
        if validate:
            ax.plot(self.val_percent_loops_plot, label='test', linewidth=0.2)
        ax.legend(loc='best', prop={'size': 4})

        plt.tight_layout(0.5)
        plt.subplots_adjust(top=0.8)
        frame_num = "{:04d}".format(epoch)
        fig.savefig("images/training_viz_%s.png" % frame_num, dpi=300)
        plt.close()

    @staticmethod
    def total_loops_metric(ground_truth, enhanced, cutoff=1.5):
        mask = (enhanced >= cutoff)
        loops = tf.boolean_mask(K.ones_like(enhanced), mask=mask)
        return K.sum(loops)

    @staticmethod
    def recovered_loops(ground_truth, enhanced, cutoff=1.5):
        def total_loops(matrix):
            mask = (matrix >= cutoff)
            loops = tf.boolean_mask(K.ones_like(matrix), mask=mask)
            return tf.count_nonzero(loops)

        def total_overlap(a, b):
            mask_a = (a >= cutoff)
            mask_b = (b >= cutoff)
            mask = tf.logical_and(mask_a, mask_b)
            loops = tf.boolean_mask(K.ones_like(a), mask=mask)
            return tf.count_nonzero(loops)
        total = total_loops(ground_truth)
        overlap = total_overlap(enhanced, ground_truth)
        return 0 if total == 0 else overlap / total
