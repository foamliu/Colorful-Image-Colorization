import os

import cv2 as cv
import numpy as np
import sklearn.neighbors as nn
from keras.utils import Sequence

from config import batch_size, img_rows, img_cols, nb_neighbors

train_images_folder = 'data/instance-level_human_parsing/Training/Images'
valid_images_folder = 'data/instance-level_human_parsing/Validation/Images'


def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    ab = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    # Smooth the weights with a gaussian kernel
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    # format the target
    y = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts
    y = y.reshape(h, w, nb_q)
    return y


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            id_file = 'data/instance-level_human_parsing/Training/train_id.txt'
            self.images_folder = train_images_folder
        else:
            id_file = 'data/instance-level_human_parsing/Validation/val_id.txt'
            self.images_folder = valid_images_folder

        with open(id_file, 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.names)

        # Load the array of quantized ab value
        q_ab = np.load("data/pts_in_hull.npy")
        self.nb_q = q_ab.shape[0]
        # Fit a NN to q_ab
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        out_img_rows, out_img_cols = img_rows // 4, img_cols // 4

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 1), dtype=np.float32)
        batch_y = np.empty((length, out_img_rows, out_img_cols, self.nb_q), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            filename = os.path.join(self.images_folder, name + '.jpg')
            # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
            bgr = cv.imread(filename)
            bgr = cv.resize(bgr, (img_rows, img_cols), cv.INTER_CUBIC)
            gray = cv.imread(filename, 0)
            gray = cv.resize(gray, (img_rows, img_cols), cv.INTER_CUBIC)
            lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
            x = gray / 255.

            out_lab = cv.resize(lab, (out_img_rows, out_img_cols), cv.INTER_CUBIC)
            # Before: 42 <=a<= 226, 20 <=b<= 223
            # After: -86 <=a<= 98, -108 <=b<= 95
            out_ab = out_lab[:, :, 1:].astype(np.int32) - 128

            y = get_soft_encoding(out_ab, self.nn_finder, self.nb_q)

            if np.random.random_sample() > 0.5:
                x = np.fliplr(x)
                y = np.fliplr(y)

            batch_x[i_batch, :, :, 0] = x
            batch_y[i_batch] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
