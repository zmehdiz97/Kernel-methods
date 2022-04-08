import os

import matplotlib.pyplot as plt
import numpy as np


def load_images(path):
    """ Read a csv file from path and returns a numpy array """
    # Check if we have a .npy (numpy) version of the images (faster)
    path_npy = path.replace(".csv", ".npy")
    if os.path.exists(path_npy):
        images = np.load(path_npy)
    else:
        print(f'saving images in .npy format to {path_npy} ...')
        images = np.genfromtxt(path, delimiter=",")
        # A trailing comma adds one pixel, remove it
        n_pixels = images.shape[1] - 1
        images = images[:, :n_pixels]
        np.save(path_npy, images)
    return images


def deflatten(X):
    """
    Takes images of shape (n_samples, n_pixels) or (n_pixels,) and reshape to
    (n_samples, width, height, 3) with widht = height
    """
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    n_samples, n_pixels = X.shape
    width = int(np.sqrt(n_pixels // 3))
    assert n_pixels == 3 * width * width
    X = X.reshape((n_samples, 3, width, width))
    X = X.transpose((0, 2, 3, 1))
    return X


def flatten(X):
    """
    Takes images of shape (n_samples, width, height, 3) and reshape to
    (n_samples, width * height * 3)
    """
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=0)
    n_samples, width, height, n_channels = X.shape
    assert n_channels == 3
    assert width == height
    X = X.transpose((0, 3, 1, 2))
    X = X.reshape((n_samples, width * height * n_channels))
    return X


def plot_image(img):
    if len(img.shape) == 1:
        img = deflatten(img)[0]
    assert len(img.shape) == 3
    # Convert centered values to ints between 0 and 255
    img = ((img - np.min(img))/(img.max() - img.min()) * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
