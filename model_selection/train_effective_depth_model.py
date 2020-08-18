import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from model_selection.multiple_depths_data_generator import DataGenerator
from model_selection.depth_model_callback import VizCallback

np.random.seed(36)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--anchor_dir', required=True, type=str)
parser.add_argument('--matrix_size', required=False, type=int, default=512)
parser.add_argument('--step_size', required=False, type=int, default=64)
parser.add_argument('--depth', required=False, type=int, default=4)
parser.add_argument('--filters', required=False, type=int, default=4)
parser.add_argument('--n_epochs', required=False, type=int, default=5)
args = parser.parse_args()

data_dir = args.data_dir
anchor_dir = args.anchor_dir
matrix_size = args.matrix_size
step_size = args.step_size
depth = args.depth
filters = args.filters
n_epochs = args.n_epochs
max_depth = 5000000

data_generator = DataGenerator(data_dir, anchor_dir, matrix_size, step_size, diagonal_only=True, max_depth=max_depth)
test_generator = DataGenerator(data_dir, anchor_dir, matrix_size, step_size, diagonal_only=True, max_depth=max_depth, test=True)
print('Input shape:', data_generator.input_shape)


input = keras.layers.Input(data_generator.input_shape)
x = input

feat_maps = []

for i in range(depth):
    x = keras.layers.Conv2D(2 ** i * filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    feat_maps.append(x)
    #x = keras.layers.MaxPooling2D((2, 2))(x)

#x = keras.layers.Flatten()(x)
#x = keras.layers.Dense(512, activation='relu')(x)
#x = keras.layers.Dense(256, activation='relu')(x)
#x = keras.layers.Dense(128, activation='relu')(x)
#x = keras.layers.Dense(1, activation='relu')(x)
x = keras.layers.concatenate(feat_maps)
x = keras.layers.GlobalAveragePooling2D()(x)
#out = keras.layers.Dense(len(data_generator.read_depth_labels), activation='softmax')(x)
out = keras.layers.Dense(1, activation='relu')(x)

model = keras.models.Model(input, out)
'''
inputs = keras.layers.Input(data_generator.input_shape, dtype=tf.uint8)
inputs = tf.cast(inputs, tf.float32)
inputs = tf.keras.applications.mobilenet.preprocess_input(inputs)
model = ResNet50V2(include_top=True,
                   weights=None,
                   input_tensor=inputs,
                   classes=len(data_generator.read_depth_labels))
                   '''
print(model.summary())
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
'''
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       'categorical_hinge',
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.FalsePositives(),
                       tf.keras.metrics.Precision(name='precision_top1', top_k=1),
                       tf.keras.metrics.Precision(name='precision_top3', top_k=3),
                       tf.keras.metrics.Recall(name='recall_top1', top_k=1),
                       tf.keras.metrics.Recall(name='recall_top3', top_k=3), ])
'''

checkpoint = ModelCheckpoint("model_selector_checkpoint", monitor='accuracy', verbose=0, save_best_only=False,
                             save_weights_only=False, mode='auto', save_freq='epoch')

#viz_callback = VizCallback(model, data_generator)

# Define the Keras TensorBoard callback.
logdir = 'logs/'
os.makedirs(logdir, exist_ok=True)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)

history = model.fit_generator(data_generator.generate_batches(),
                              steps_per_epoch=int(data_generator.steps_per_epoch / 16),
                              validation_data=test_generator.generate_batches(),
                              validation_steps=int(data_generator.steps_per_epoch / 16),
                              epochs=n_epochs,
                              verbose=1,
                              callbacks=[tensorboard_callback])

# model.fit_generator(data_generator, epochs=n_epochs, verbose=2)
model.save('model_selector.h5')

loss = history.history['loss']
#acc = history.history['accuracy']
val_loss = history.history['val_loss']
#val_acc = history.history['val_accuracy']

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(loss, label='train')
axs[0].plot(val_loss, label='test')
axs[0].legend(loc='best')
#axs[1].plot(acc, label='train')
#axs[1].plot(val_acc, label='test')
axs[1].legend(loc='best')

plt.show()
