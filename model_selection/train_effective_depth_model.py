import argparse
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Input
from keras.models import Model
from keras.optimizers import Adam

from model_selection.multiple_depths_data_generator import DataGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--anchor_dir', required=True, type=str)
parser.add_argument('--matrix_size', required=False, type=int, default=128)
parser.add_argument('--step_size', required=False, type=int, default=64)
parser.add_argument('--depth', required=False, type=int, default=4)
parser.add_argument('--filters', required=False, type=int, default=4)
parser.add_argument('--n_epochs', required=False, type=int, default=1000)
args = parser.parse_args()

data_dir = args.data_dir
anchor_dir = args.anchor_dir
matrix_size = args.matrix_size
step_size = args.step_size
depth = args.depth
filters = args.filters
n_epochs = args.n_epochs

data_generator = DataGenerator(data_dir, anchor_dir, matrix_size, step_size)

input = Input(data_generator.input_shape)
x = input

for i in range(depth):
    x = Conv2D(i * filters, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D()(x)

x = Dense(128, activation='relu')
out = Dense(1, activation='relu')

model = Model(input, out)
model.compile(optimizer=Adam(), loss='mse')

model.fit_generator(data_generator, epochs=n_epochs, verbose=1)