import os

import cv2 as cv
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import sklearn.neighbors as nn
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve


def load_data(size=64):
    image_folder = '/mnt/code/ImageNet-Downloader/image/resized'
    names = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
    np.random.shuffle(names)
    num_samples = 100000
    X_ab = np.empty((num_samples, size, size, 2))
    for i in range(num_samples):
        name = names[i]
        filename = os.path.join(image_folder, name)
        bgr = cv.imread(filename)
        bgr = cv.resize(bgr, (size, size), cv.INTER_CUBIC)
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        lab = lab.astype(np.int32)
        X_ab[i] = lab[:, :, 1:] - 128
    return X_ab


def compute_color_prior(X_ab, size=64, do_plot=False):
    # Load the gamut points location
    q_ab = np.load(os.path.join(data_dir, "pts_in_hull.npy"))

    if do_plot:
        plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        for i in range(q_ab.shape[0]):
            ax.scatter(q_ab[:, 0], q_ab[:, 1])
            ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
            ax.set_xlim([-110, 110])
            ax.set_ylim([-110, 110])

    npts, c, h, w = X_ab.shape
    X_a = np.ravel(X_ab[:, :, :, 0])
    X_b = np.ravel(X_ab[:, :, :, 1])
    X_ab = np.vstack((X_a, X_b)).T

    if do_plot:
        plt.hist2d(X_ab[:, 0], X_ab[:, 1], bins=100, normed=True, norm=LogNorm())
        plt.xlim([-110, 110])
        plt.ylim([-110, 110])
        plt.colorbar()
        plt.show()
        plt.clf()
        plt.close()

    # Create nearest neighbord instance with index = q_ab
    NN = 1
    nearest = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(q_ab)
    # Find index of nearest neighbor for X_ab
    dists, ind = nearest.kneighbors(X_ab)

    # We now count the number of occurrences of each color
    ind = np.ravel(ind)
    counts = np.bincount(ind)
    idxs = np.nonzero(counts)[0]
    prior_prob = np.zeros((q_ab.shape[0]))
    for i in range(q_ab.shape[0]):
        prior_prob[idxs] = counts[idxs]

    # We turn this into a color probability
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Save
    np.save(os.path.join(data_dir, "prior_prob.npy"), prior_prob)

    if do_plot:
        plt.hist(prior_prob, bins=100)
        plt.yscale("log")
        plt.show()


def smooth_color_prior(size=64, sigma=5, do_plot=False):
    prior_prob = np.load(os.path.join(data_dir, "prior_prob.npy"))
    # add an epsilon to prior prob to avoid 0 vakues and possible NaN
    prior_prob += 1E-3 * np.min(prior_prob)
    # renormalize
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Smooth with gaussian
    f = interp1d(np.arange(prior_prob.shape[0]), prior_prob)
    xx = np.linspace(0, prior_prob.shape[0] - 1, 1000)
    yy = f(xx)
    window = gaussian(2000, sigma)  # 2000 pts in the window, sigma=5
    smoothed = convolve(yy, window / window.sum(), mode='same')
    fout = interp1d(xx, smoothed)
    prior_prob_smoothed = np.array([fout(i) for i in range(prior_prob.shape[0])])
    prior_prob_smoothed = prior_prob_smoothed / np.sum(prior_prob_smoothed)

    # Save
    file_name = os.path.join(data_dir, "prior_prob_smoothed.npy")
    np.save(file_name, prior_prob_smoothed)

    if do_plot:
        plt.plot(prior_prob)
        plt.plot(prior_prob_smoothed, "g--")
        plt.plot(xx, smoothed, "r-")
        plt.yscale("log")
        plt.show()


def compute_prior_factor(size=64, gamma=0.5, alpha=1, do_plot=False):
    file_name = os.path.join(data_dir, "prior_prob_smoothed.npy")
    prior_prob_smoothed = np.load(file_name)

    u = np.ones_like(prior_prob_smoothed)
    u = u / np.sum(1.0 * u)

    prior_factor = (1 - gamma) * prior_prob_smoothed + gamma * u
    prior_factor = np.power(prior_factor, -alpha)

    # renormalize
    prior_factor = prior_factor / (np.sum(prior_factor * prior_prob_smoothed))

    file_name = os.path.join(data_dir, "prior_factor.npy")
    np.save(file_name, prior_factor)

    if do_plot:
        plt.plot(prior_factor)
        plt.yscale("log")
        plt.show()


if __name__ == '__main__':
    data_dir = 'data/'
    do_plot = True

    X_ab = load_data()
    compute_color_prior(X_ab, do_plot=True)
    smooth_color_prior(do_plot=True)
    compute_prior_factor(do_plot=True)
