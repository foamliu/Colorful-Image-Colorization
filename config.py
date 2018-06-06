import scipy.io
import numpy as np

img_rows, img_cols = 256, 256
channel = 3
batch_size = 96
epochs = 1000
patience = 50
num_train_samples = 28280
num_valid_samples = 5000
num_classes = 313
epsilon = 1e-6
epsilon_sqr = epsilon ** 2

mat = scipy.io.loadmat('human_colormap.mat')
color_map = (mat['colormap'] * 256).astype(np.int32)

nb_neighbors = 5
# temperature parameter T
T = .14
