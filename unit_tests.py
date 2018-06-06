import os
import random
import unittest

import cv2 as cv
import numpy as np
import sklearn.neighbors as nn

from data_generator import get_soft_encoding


class TestStringMethods(unittest.TestCase):

    def test_get_soft_encoding(self):
        nb_neighbors = 5
        # Load the array of quantized ab value
        q_ab = np.load("data/pts_in_hull.npy")
        nb_q = q_ab.shape[0]
        # Fit a NN to q_ab
        nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

        id_file = 'data/instance-level_human_parsing/Training/train_id.txt'
        images_folder = 'data/instance-level_human_parsing/Training/Images'
        with open(id_file, 'r') as f:
            names = f.read().splitlines()
        name = random.choice(names)
        filename = os.path.join(images_folder, name + '.jpg')
        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr = cv.imread(filename)
        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        y = get_soft_encoding(lab[:, :, 1:], nn_finder, nb_q)


if __name__ == '__main__':
    unittest.main()
