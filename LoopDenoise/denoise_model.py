import os
import glob
import datetime
import time
import pickle
from keras.optimizers import Adam
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from cnn_architectures import Autoencoder
from utils import generate_batches_from_chr, save_samples, draw_heatmap, get_model_memory_usage
from utils import save_images_to_video, normalize_matrix, denormalize_matrix


class DenoiseModel(Autoencoder):
    """
    Class used to create a denoise model which can be trained by calling the ``train`` function.
    """
    def __init__(self,
                 matrix_size=128,
                 step_size=64,
                 batch_size=64,
                 epochs=100,
                 load_pretrained_epoch=0,
                 steps_per_checkpoint=10,
                 steps_per_model_checkpoint=1,
                 start_filters=16,
                 filter_size=3,
                 transpose_filter_size=2,
                 branching_rate=2.,
                 depth=2,
                 dropout=0.5,
                 normalize=False,
                 model_name='denoise',
                 activation='relu',
                 loss_type='mse',
                 model_out='models/denoise_model/',
                 verbose=True):
        """
        Initialize the denoise model object using the given parameters.

        Args:
            matrix_size (:obj:`int`) : number of anchors to use for each input sample
            step_size (:obj:`int`) : size of steps used when generating batches.  Values less than ``matrix size`` will include overlapping regions
            batch_size (:obj:`int`) : number of samples to use in each training batch
            epochs (:obj:`int`) : number of epochs until stopping training process
            load_pretrained_epoch (:obj:`int`) : load a previous model checkpoint from a specific epoch
            steps_per_checkpoint (:obj:`int`) : number of batches between saving training visualizations
            steps_per_model_checkpoint (:obj:`int`) : number of epochs between saving a model checkpoint
            start_filters (:obj:`int`) : number of filters in the first layer of the network
            ilter_size (:obj:`int`): size of filters in each convolution layer, except the final linear convolution layer in the U-Net architecture
            branching_rate (:obj:`int`) : multiplier for the number of filters in each successive layer of the U-net model
            depth (:obj:`int`) : number of layers in the U-Net model
            dropout (:obj:`float`) : probability of randomly dropping certain network connection during training
            train_on_pairs (:obj:`bool`) : set to True to train the model by each pairwise combination of replicates
            normalize (:obj:`bool`) : set to True to normalize inputs between ``[0, 1]`` before training
            model_name (:obj:`str`) : name used to identify model files and training visualization labels
            activation (:obj:`str`) : activation function used after each convolutional layer
            loss_type (:obj:`str`) : loss function which will be minimized during the training process
            model_out (:obj:`str`) : directory to store the trained model checkpoints
            verbose (:obj:`bool`) : set to True to display training statistics after each batch
        """
        super().__init__(matrix_size=matrix_size,
                         step_size=step_size,
                         start_filters=start_filters,
                         filter_size=filter_size,
                         transpose_filter_size=transpose_filter_size,
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
        self.loss_name = loss_type
        self.jaccard_plot = np.array([])
        if loss_type == 'jaccard':  # jaccard is not a built-in loss function
            self.loss_type = self.jaccard_metric_weighted(smooth=100)  # custom weighted jaccard method
        else:
            self.loss_type = loss_type  # otherwise we can keep using the string loss type
        self.model_out = model_out
        self.verbose = verbose
        if load_pretrained_epoch > 0:
            model_params = self.model_params_to_name()
            with open(self.model_out + "n2n_" + self.model_name + model_params + '.json', 'r') as f:
                self.n2n = model_from_json(f.read())
            # Load weights into the new model
            self.n2n.load_weights(self.model_out + 'n2n_' + self.model_name + model_params + '_%depochs.h5' % load_pretrained_epoch)
            print(self.n2n.summary())
        else:
            self.n2n = self.get_autoencoder()

    def train(self,
              noisy_dir,
              target_dir,
              anchor_dir,
              multi_input=False,
              validate=False,
              val_noisy_dir=None,
              val_target_dir=None,
              val_anchor_dir=None,
              val_multi_input=False,
              learning_rate=1e-4,
              save_imgs=False):
        """
        Train the denoise model using the anchor to anchor files in the provided directories

        Args:
            noisy_dir (:obj:`str`) : directory containing all noisy anchor to anchor replicate files of each chromosome to be used as inputs to the denoise model
            target_dir (:obj:`str`) : directory containing target anchor to anchor files for each chromosome
            anchor_dir (:obj:`str`) : directory containing the ``.bed`` anchor reference files for each chromosome
            multi_input (:obj:`bool`) : set to True if ``noisy_dir`` contains multiple folders of multiple replicates, otherwise just use a single replicate for training
            validate (:obj:`bool`) : set to True to validate the model during training (slower since we have test the model on a full replicate set after each epoch)
            val_noisy_dir (:obj:`str`) : directory containing all noisy anchor to anchor replicate files of each chromosome to be used for validation
            val_target_dir (:obj:`str`) : directory containing target anchor to anchor files for each chromosome to be used for validation
            val_anchor_dir (:obj:`str`) : directory containing the ``.bed`` anchor reference files for each chromosome to be used for validation
            val_multi_input (:obj:`bool`) : set to True if ``val_noisy_dir`` contains multiple folders of multiple replicates
            learning_rate (:obj:`float`) : size of each gradient step during mini-batch optimization
            save_imgs (:obj:`bool`) : set to True to save training visualizations
        """
        optimizer = Adam(lr=learning_rate)
        metrics = [self.psnr, self.jaccard_metric_weighted(smooth=100)]  # called and return after train_on_batch
        self.n2n.compile(optimizer=optimizer,
                         loss=self.loss_type,
                         metrics=metrics)
        print('Memory Estimates:')
        print(get_model_memory_usage(self.batch_size, self.n2n), 'GB\t')
        try:
            files = glob.glob('images/*')
            for f in files:
                os.remove(f)  # remove any previously saved training visualizations
        except Exception as e:
            pass
        if save_imgs:
            try:
                os.mkdir('images/')
            except FileExistsError as e:
                pass
            # get sample replicates to be used in training visualizations
            save_samples(noisy_dir,
                         target_dir,
                         matrix_size=self.matrix_size,
                         multi_input=True,
                         anchor_dir=anchor_dir,
                         locus=4204)
            sample_1 = np.load('data/sample_1.npy')
            sample_2 = np.load('data/sample_2.npy')
            sample_3 = np.load('data/sample_3.npy')
            sample_target = np.load('data/sample_combined.npy')

        step_i = 0  # current batch number of current epoch
        plot_i = 0  # current number of training visualizations saved so far
        start_time = time.time()
        if self.load_pretrained_epoch > 0:
            self.loss_plot = np.load('results/saved_curves/loss.npy')[:self.load_pretrained_epoch]
            self.psnr_plot = np.load('results/saved_curves/psnr.npy')[:self.load_pretrained_epoch]
            self.jaccard_plot = np.load('results/saved_curves/jaccard.npy')[:self.load_pretrained_epoch]
            if validate:
                self.val_loss_plot = np.load('results/saved_curves/val_loss.npy')[:self.load_pretrained_epoch]
                self.val_psnr_plot = np.load('results/saved_curves/val_psnr.npy')[:self.load_pretrained_epoch]
                self.val_jaccard_plot = np.load('results/saved_curves/val_jaccard.npy')[:self.load_pretrained_epoch]
        for epoch_i in range(self.load_pretrained_epoch, self.epochs + self.load_pretrained_epoch):
            batch_loss = np.array([])
            batch_psnr = np.array([])
            batch_jaccard = np.array([])
            for noisy_batch, target_batch, _ in generate_batches_from_chr(noisy_dir,
                                                                          target_dir,
                                                                          self.matrix_size,
                                                                          self.batch_size,
                                                                          shuffle=True,
                                                                          step_size=self.step_size,
                                                                          random_steps=True,
                                                                          multi_input=multi_input,
                                                                          anchor_dir=anchor_dir,
                                                                          normalize=self.normalize):
                step_start_time = time.time()
                loss = self.n2n.train_on_batch(noisy_batch, target_batch)
                batch_loss = np.append(batch_loss, loss[0])
                batch_psnr = np.append(batch_psnr, loss[1])
                batch_jaccard = np.append(batch_jaccard, loss[2])
                if self.verbose:
                    print("%d-%d %ds [Loss: %.3f][PSNR: %.3f, Jaccard: %.3f]" %
                          (plot_i,
                           step_i,
                           time.time() - step_start_time,
                           loss[0],
                           loss[1],
                           loss[2]
                           ))
            self.loss_plot = np.append(self.loss_plot, batch_loss.mean())
            self.psnr_plot = np.append(self.psnr_plot, batch_psnr.mean())
            self.jaccard_plot = np.append(self.jaccard_plot, batch_jaccard.mean())
            if validate:
                val_loss = np.array([])
                val_psnr = np.array([])
                val_jaccard = np.array([])
                for val_noisy_batch, val_target_batch, _ in generate_batches_from_chr(val_noisy_dir,
                                                                                      val_target_dir,
                                                                                      self.matrix_size,
                                                                                      self.batch_size,
                                                                                      multi_input=val_multi_input,
                                                                                      shuffle=True,
                                                                                      step_size=self.step_size,
                                                                                      random_steps=True,
                                                                                      normalize=self.normalize,
                                                                                      anchor_dir=val_anchor_dir):
                    val_metrics = self.enhance_model.test_on_batch(val_noisy_batch, val_target_batch)
                    val_loss = np.append(val_loss, val_metrics[0])
                    val_psnr = np.append(val_psnr, val_metrics[1])
                    val_jaccard = np.append(val_jaccard, val_metrics[2])

                self.val_loss_plot = np.append(self.val_loss_plot, np.mean(val_loss))
                self.val_psnr_plot = np.append(self.val_psnr_plot, np.mean(val_psnr))
                self.val_jaccard_plot = np.append(self.val_jaccard_plot, np.mean(val_jaccard))
            if epoch_i % self.steps_per_model_checkpoint == 0:
                print('%d - Loss: %.4f, PSNR: %.2f, Weighted Jaccard: %.4f' % (epoch_i,
                                                                               self.loss_plot[-1],
                                                                               self.psnr_plot[-1],
                                                                               self.jaccard_plot[-1]))
                if validate:
                    print('\tTest Loss: %.4f, Test PSNR: %.2f, Test Weighted Jaccard: %.4f' % (self.val_loss_plot[-1],
                                                                                               self.val_psnr_plot[-1],
                                                                                               self.val_jaccard_plot[-1]))
                self.save_epoch_curves(epoch_i)
                self.save_model(epoch_i)
                self.plot_training_visualization(start_time=start_time,
                                                 epoch=plot_i,
                                                 matrix_a=sample_1,
                                                 matrix_b=sample_2,
                                                 matrix_c=sample_3,
                                                 target=sample_target,
                                                 validate=validate)
                plot_i += 1

        self.save_model(epoch_i)
        self.save_results(save_imgs=save_imgs)

    def save_results(self, save_imgs=False):
        out_dir = 'results/denoise/model_experiments/' + self.model_name + '/'
        os.makedirs(out_dir, exist_ok=True)
        if save_imgs:
            save_images_to_video(self.model_name, out_dir)
        np.save(out_dir + 'loss.npy', self.loss_plot)
        np.save(out_dir + 'psnr.npy', self.psnr_plot)
        np.save(out_dir + 'jaccard.npy', self.jaccard_plot)
        self.n2n.save(out_dir + 'model_checkpoint_%d_epochs.h5' % self.epochs)
        self.n2n.save_weights(out_dir + self.model_name + '.h5')
        with open(out_dir + self.model_name + '.json', 'w') as f:
            f.write(self.n2n.to_json())

    def save_model(self, epoch):
        """
        Saves the current state of the model at a certain number of epochs.

        Args:
            epoch (:obj:`int`) : current epoch of the training process
        """
        # Save the models
        try:
            os.mkdir(self.model_out)
        except FileExistsError:
            pass
        self.n2n.save_weights(self.model_out + self.model_name + '_%depochs.h5' % epoch)
        with open(self.model_out + self.model_name + '.json', 'w') as f:
            f.write(self.n2n.to_json())

    def save_epoch_curves(self, epoch_i, validate=False):
        """
        Saves the arrays and figures containing loss, PSNR, and jaccard plots.

        Args:
            epoch_i (:obj:`int`) : current epoch of the training process
            validate (:obj:`bool`) : set to True to plot validation curves on top of training curves
        """
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
        np.save('results/saved_curves/jaccard.npy', self.jaccard_plot)

        fig = plt.figure(figsize=(24, 8))

        ax = plt.subplot(131)
        ax.plot(self.loss_plot, label='train')
        if validate:
            ax.plot(self.val_loss_plot, label='test')
        ax.set_title('Loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel(self.loss_name)
        ax.legend(loc='best')

        ax = plt.subplot(132)
        ax.plot(self.psnr_plot, label='train')
        if validate:
            ax.plot(self.val_psnr_plot, label='test')
        ax.set_title('PSNR')
        ax.set_xlabel('epoch')
        ax.set_ylabel('dB')
        ax.legend(loc='best')

        ax = plt.subplot(133)
        ax.plot(self.jaccard_plot, label='train')
        if validate:
            ax.plot(self.val_jaccard_plot, label='test')
        ax.set_title('Weighted Jaccard')
        ax.set_xlabel('epoch')
        ax.set_ylabel('distance')
        ax.legend(loc='best')

        fig.savefig('results/training_curves/epoch_%d.png' % epoch_i)
        plt.close()

    def plot_training_visualization(self,
                                    start_time,
                                    epoch,
                                    matrix_a,
                                    matrix_b,
                                    matrix_c,
                                    target,
                                    validate=True):
        """
        Generate and save a training visualization showing example heatmaps and training curves from both training and validation sets.
        Args:
            start_time (:obj:`int`) : start time of the training process in seconds
            epoch (:obj:`int`) : current epoch of the training process
            matrix_a (:obj:`numpy.array`) : sample input matrix with shape ``[1, matrix_size, matrix_size, 1]``
            matrix_b (:obj:`numpy.array`) : sample input matrix
            matrix_c (:obj:`numpy.array`) : sample input matrix
            target (:obj:`numpy.array`) : sample target matrix
            validate (:obj:`bool`) : set to True to include validation training curves
        """
        if self.normalize:
            batch_1 = normalize_matrix(matrix_a)
            batch_2 = normalize_matrix(matrix_b)
            batch_3 = normalize_matrix(matrix_c)
        else:
            batch_1 = matrix_a
            batch_2 = matrix_b
            batch_3 = matrix_c
        denoised_a = self.n2n.predict(batch_1)
        denoised_b = self.n2n.predict(batch_2)
        denoised_c = self.n2n.predict(batch_3)
        denoised_a = (denoised_a + denoised_a.T) * 0.5
        denoised_b = (denoised_b + denoised_b.T) * 0.5
        denoised_c = (denoised_c + denoised_c.T) * 0.5
        if self.normalize:
            denoised_a = denormalize_matrix(denoised_a)
            denoised_b = denormalize_matrix(denoised_b)
            denoised_c = denormalize_matrix(denoised_c)
        # plot training visualization
        fig = plt.figure()

        fig.suptitle(self.model_name + ' ' + self.loss_name + '\nTrained on ' + str(epoch * self.batch_size * 2 * self.steps_per_checkpoint) + ' matrices\n' + str(
                datetime.timedelta(seconds=(int(time.time() - start_time)))), size=12)

        ax = plt.subplot(341)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.set_ylabel('Total Loops', fontsize=6)
        plt.title('Total Loops ($ratio > 2$)', size=6)
        #ax.plot(self.total_loops_plot['1'], label='1', linewidth=0.2)
        #ax.plot(self.total_loops_plot['2'], label='2', linewidth=0.2)
        #ax.plot(self.total_loops_plot['3'], label='3', linewidth=0.2)
        ax.legend(loc='best', prop={'size': 4})

        ax = plt.subplot(342)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Replicate A', size=6)
        draw_heatmap(matrix_a[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(343)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Replicate B', size=6)
        draw_heatmap(matrix_b[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(344)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        plt.title('Replicate C', size=6)
        draw_heatmap(matrix_c[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(345)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.set_ylabel(self.loss_name, fontsize=6)
        plt.title('Model Objective', size=6)
        ax.plot(self.loss_plot, label='train', linewidth=0.2)
        if validate:
            ax.plot(self.val_loss_plot, label='test', linewidth=0.2)
        ax.legend(loc='best', prop={'size': 4})

        ax = plt.subplot(346)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        ax.xaxis.tick_top()
        plt.title('Denoised A', size=6)
        draw_heatmap(denoised_a[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(347)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        ax.xaxis.tick_top()
        plt.title('Denoised B', size=6)
        draw_heatmap(denoised_b[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(348)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        ax.xaxis.tick_top()
        plt.title('Denoised C', size=6)
        draw_heatmap(denoised_c[0, ..., 0], 0, ax=ax)

        ax = plt.subplot(349)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.set_ylabel('distance', fontsize=6)
        plt.title('Weighted Jaccard', size=6)
        ax.plot(self.jaccard_plot, label='train', linewidth=0.2)
        if validate:
            ax.plot(self.val_jaccard_plot, label='test', linewidth=0.2)
        ax.legend(loc='best', prop={'size': 3})

        ax = plt.subplot(3, 4, 10)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.set_ylabel('overlap/total loops', fontsize=6)
        plt.title('Reproducibility', size=6)
        #ax.plot(self.reproducibility_plot['1-2'], label='1 -- 2', alpha=0.5, linewidth=0.2)
        #ax.plot(self.reproducibility_plot['2-1'], label='2 -- 1', alpha=0.5, linewidth=0.2)
        #ax.plot(self.reproducibility_plot['1-3'], label='1 -- 3', alpha=0.5, linewidth=0.2)
        #ax.plot(self.reproducibility_plot['3-1'], label='3 -- 1', alpha=0.5, linewidth=0.2)
        #ax.plot(self.reproducibility_plot['2-3'], label='2 -- 3', alpha=0.5, linewidth=0.2)
        #ax.plot(self.reproducibility_plot['3-2'], label='3 -- 2', alpha=0.5, linewidth=0.2)
        ax.legend(loc='best', prop={'size': 4})

        ax = plt.subplot(3, 4, 11)
        ax.yaxis.set_tick_params(labelsize=4)
        ax.xaxis.set_tick_params(labelsize=4)
        ax.set_ylabel('dB', fontsize=6)
        plt.title('PSNR', size=6)
        ax.plot(self.psnr_plot, label='train', linewidth=0.2)
        #if validate:
            #ax.plot(val_psnr, label='test', linewidth=0.2)
        ax.legend(loc='best', prop={'size': 4})

        if target is not None:
            ax = plt.subplot(3, 4, 12)
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticks([])
            plt.title('Combined Target', size=6)
            draw_heatmap(target[0, ..., 0], 0, ax=ax)

        plt.tight_layout(0.8)
        plt.subplots_adjust(top=0.8)
        frame_num = "{:04d}".format(epoch)
        fig.savefig("images/training_viz_%s.png" % frame_num, dpi=300)
        plt.close()