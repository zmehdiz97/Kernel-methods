import numpy as np
from tqdm import tqdm
from img_utils import load_images, deflatten
import os


class HOGFeatureExtractor:
    """
    nbins: number of bins that will be used
    unsigned: if True the sign of the angle is not considered
    """
    def __init__(self, nbins=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), unsigned=True):
        self.nbins = nbins
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.unsigned = unsigned


    def _calc_gradient_for_channel(self, I, unflatten):
        nX, nY = I.shape
        cx, cy = self.pixels_per_cell # pixels per cell
        histogram = np.zeros((4, 4, self.nbins))

        for i in range(0, nX):
            for j in range(0, nY):
                dx, dy = 0, 0
                if i < nX - 1:
                    dx += I[i + 1, j]
                if i > 0:
                    dx -= I[i - 1, j]
                if j < nY - 1:
                    dy += I[i, j + 1]
                if j > 0:
                    dy -= I[i, j - 1]

                if dy == 0 and dx == 0:
                    continue

                magnitude = np.sqrt(dx**2 + dy**2)
                if self.unsigned:
                    if dx == 0:
                        angle = np.pi / 2
                    else:
                        angle = np.arctan(dy / dx)
                    angle = (angle + np.pi / 2) / (np.pi / self.nbins)
                else:
                    angle = np.arctan2(dx, dy)
                    angle = (angle + np.pi) / (2 * np.pi / self.nbins)

                bin_pos = int(np.floor(angle))
                # handle corner case
                if bin_pos == self.nbins:
                    bin_pos = 0
                    angle = 0

                closest_bin = bin_pos

                if bin_pos == 0:
                    if angle < 0.5:
                        second_closest_bin = self.nbins - 1
                    else:
                        second_closest_bin = 1
                elif bin_pos == self.nbins - 1:
                    if angle < self.nbins - 0.5:
                        second_closest_bin = self.nbins - 2
                    else:
                        second_closest_bin = 0
                else:
                    if angle < bin_pos + 0.5:
                        second_closest_bin = bin_pos - 1
                    else:
                        second_closest_bin = bin_pos + 1

                # closest_bin_distance + second_closest_bin_distance = 1
                if angle < bin_pos + 0.5:
                    second_closest_bin_distance = angle - (bin_pos - 0.5)
                else:
                    second_closest_bin_distance = (bin_pos + 1.5) - angle

                r = second_closest_bin_distance
                histogram[i // cx, j // cy, closest_bin] += r * magnitude
                histogram[i // cx, j // cy, second_closest_bin] += (1 - r) * magnitude
        
        n_cellsx = int(np.floor(nX / cx))  # number of cells in x
        n_cellsy = int(np.floor(nY / cy))  # number of cells in y
        
        blockx, blocky = self.cells_per_block[0], self.cells_per_block[1]         
        new_size_x = n_cellsx - blockx + 1
        new_size_y = n_cellsy - blocky + 1
        ret = np.zeros((new_size_x, new_size_y, self.nbins * blockx * blocky))

        for i in range(new_size_x):
            for j in range(new_size_y):
                aux = histogram[i:i + blockx, j:j + blocky, :].flatten().copy()
                aux = aux / np.linalg.norm(aux)
                ret[i, j, :] = aux

        if unflatten:
            ret.reshape(new_size_x*new_size_y, -1)
        return ret.flatten()

    def _calc_gradient_for_image(self, I, unflatten):
        nchannels = I.shape[2]
        ret = []

        for i in range(nchannels):
            ret.append(self._calc_gradient_for_channel(I[:,:,i], unflatten))

        if unflatten:
            return np.array(ret).reshape(nchannels * ret[0].shape[0], -1)
        return np.array(ret).flatten()

    def predict(self, X, unflatten=False):
        assert X.ndim == 4
        print("Extracting HOG features")
        n = X.shape[0]
        ret = []

        for i in tqdm(range(n)):
            ret.append(self._calc_gradient_for_image(X[i,:,:,:], unflatten))

        return np.array(ret)
    
if __name__ == '__main__':
    input_path = 'data'
    input_file = os.path.join(input_path, 'Xtr.csv') # 'Xtr.csv' or 'Xte.csv'
    imgs = load_images(input_file)
    imgs = deflatten(imgs)
    n_img = imgs.shape[0]
        
    HOG = HOGFeatureExtractor(nbins=9)
    feat = HOG.predict(imgs)    
    print(feat.shape)
    # output setups
    output_path = 'data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_file = os.path.join(output_path, 'Xtr_hoog.npy') 
    np.save(output_file, feat)