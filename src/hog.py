import matplotlib
import numpy as np
import os
from tqdm import tqdm
from skimage.feature import hog

from img_utils import load_images, deflatten
def rgb2gray(rgb):
    """
    Convert RGB image to grayscale
    Parameters:
        rgb : RGB image
    
    Returns:
        gray : grayscale image
  
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

class HOGFeatureExtractor:
    """
    nbins: number of bins that will be used
    """
    def __init__(self, nbins=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.nbins = nbins
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
    def extract_hog(self, im, gray_scale_fetat=False):
        """
        Extract Histogram of Gradient (HOG) features for an image
        
        Inputs:
            im : an input grayscale or rgb image
        
        Returns:
            feat: Histogram of Gradient (HOG) feature
        """
        # convert rgb to grayscale
        if gray_scale_fetat:
            image = rgb2gray(im)
            return self._extract_hog(image)
        else:
            r = self._extract_hog(im[:,:,0])
            g = self._extract_hog(im[:,:,1])
            b = self._extract_hog(im[:,:,2])
            return np.concatenate((r, g, b), axis=-1)
        
    def _extract_hog(self, image):
        """
        Extract Histogram of Gradient (HOG) features for an image
        
        Inputs:
            im : an input grayscale or rgb image
        
        Returns:
            feat: Histogram of Gradient (HOG) feature
        """
        sx, sy = image.shape # image size
        cx, cy = self.pixels_per_cell # pixels per cell

        gx = np.zeros(image.shape)
        gy = np.zeros(image.shape)
        gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
        gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
        grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
        
        grad_dir = np.arctan(gy/(gx+ 1e-15))
        grad_dir = np.rad2deg(grad_dir) + 90
        #grad_dir = grad_dir%180
        
        n_cellsx = int(np.floor(sx / cx))  # number of cells in x
        n_cellsy = int(np.floor(sy / cy))  # number of cells in y
        
        orientation_histogram = np.zeros((n_cellsx, n_cellsy, self.nbins))
        for i in range(n_cellsx):
            for j in range(n_cellsy):
                cell_direction = grad_dir[i:i+cx, j:j+cy]
                cell_magnitude = grad_mag[i:i+cx, j:j+cy]

                orientation_histogram[i,j,:] = self.HOG_cell_histogram(cell_direction, cell_magnitude)
        
        blockx, blocky = self.cells_per_block[0], self.cells_per_block[1]         
        new_size_x = n_cellsx - blockx + 1
        new_size_y = n_cellsy - blocky + 1
        ret = np.zeros((new_size_x, new_size_y, self.nbins * blockx * blocky))
        for i in range(new_size_x):
            for j in range(new_size_y):
                aux = orientation_histogram[i:i + blockx, j:j + blocky, :].flatten().copy()
                aux = aux / np.linalg.norm(aux)
                ret[i, j, :] = aux
      
        feats = ret.flatten()  
             
        return feats

    def HOG_cell_histogram(self, cell_direction, cell_magnitude):
        assert cell_direction.shape == cell_magnitude.shape
        assert cell_direction.shape ==  self.pixels_per_cell
        
        HOG_cell_hist = np.zeros(shape=(self.nbins))
        cell_size = cell_direction.shape[0]
        cell_direction = cell_direction /(180/self.nbins)
        dir_pos = np.floor(cell_direction).astype(int)
        for row_idx in range(cell_size):
            for col_idx in range(cell_size):
                curr_direction = cell_direction[row_idx, col_idx]
                curr_magnitude = cell_magnitude[row_idx, col_idx]
                curr_pos_dir = dir_pos[row_idx, col_idx]
                if curr_pos_dir == self.nbins:
                    curr_pos_dir = 0
                    curr_direction = 0
                closest_bin = curr_pos_dir
                if curr_pos_dir == 0:
                    if curr_direction < 0.5:
                        second_closest_bin = self.nbins - 1
                    else:
                        second_closest_bin = 1
                elif curr_pos_dir == self.nbins - 1:
                    if curr_direction < self.nbins - 0.5:
                        second_closest_bin = self.nbins - 2
                    else:
                        second_closest_bin = 0
                else:
                    if curr_direction < curr_pos_dir + 0.5:
                        second_closest_bin = curr_pos_dir - 1
                    else:
                        second_closest_bin = curr_pos_dir + 1
                if curr_direction < curr_pos_dir + 0.5:
                    second_closest_bin_distance = curr_direction - (curr_pos_dir - 0.5)
                else:
                    second_closest_bin_distance = (curr_pos_dir + 1.5) - curr_direction
                
                r = second_closest_bin_distance
                HOG_cell_hist[closest_bin] += r * curr_magnitude
                HOG_cell_hist[second_closest_bin] += (1 - r) * curr_magnitude

        return HOG_cell_hist
    
    #def HOG_cell_histogram(self, cell_direction, cell_magnitude):
    #    assert cell_direction.shape == cell_magnitude.shape
    #    assert cell_direction.shape ==  self.pixels_per_cell
    #    
    #    HOG_cell_hist = np.zeros(shape=(self.nbins))
    #    cell_size = cell_direction.shape[0]
    #    hist_bins = np.arange(0,180,180/self.nbins) + 90/self.nbins
    #    for row_idx in range(cell_size):
    #        for col_idx in range(cell_size):
    #            curr_direction = cell_direction[row_idx, col_idx]
    #            curr_magnitude = cell_magnitude[row_idx, col_idx]
    #    
    #            diff = np.abs(curr_direction - hist_bins)
    #            if curr_direction < hist_bins[0]:
    #                first_bin_idx = 0
    #                second_bin_idx = hist_bins.size-1
    #            elif curr_direction > hist_bins[-1]:
    #                first_bin_idx = hist_bins.size-1
    #                second_bin_idx = 0
    #            else:
    #                first_bin_idx = np.where(diff == np.min(diff))[0][0]
    #                temp = hist_bins[[(first_bin_idx-1)%hist_bins.size, (first_bin_idx+1)%hist_bins.size]]
    #                temp2 = np.abs(curr_direction - temp)
    #                res = np.where(temp2 == np.min(temp2))[0][0]
    #                if res == 0 and first_bin_idx != 0:
    #                    second_bin_idx = first_bin_idx-1
    #                else:
    #                    second_bin_idx = first_bin_idx+1
    #            
    #            first_bin_value = hist_bins[first_bin_idx]
    #            second_bin_value = hist_bins[second_bin_idx]
    #            HOG_cell_hist[first_bin_idx] = HOG_cell_hist[first_bin_idx] + (np.abs(curr_direction - first_bin_value)/(180.0/hist_bins.size)) * curr_magnitude
    #            HOG_cell_hist[second_bin_idx] = HOG_cell_hist[second_bin_idx] + (np.abs(curr_direction - second_bin_value)/(180.0/hist_bins.size)) * curr_magnitude
    #        
    #    return HOG_cell_hist

if __name__ == '__main__':
    input_path = 'data'
    input_file = os.path.join(input_path, 'Xte.csv') # 'Xtr.csv' or 'Xte.csv'
    imgs = load_images(input_file)
    imgs = deflatten(imgs)
    n_img = imgs.shape[0]
    
    HOG = HOGFeatureExtractor(nbins=9, pixels_per_cell=(8,8))
    feat = HOG.extract_hog(imgs[0])
    X_hog = np.zeros((n_img, feat.shape[0]))
    
    # Extract features for the rest of the images
    for i in tqdm(range(n_img)):
        feat = HOG.extract_hog(imgs[i])
        X_hog[i, :] = feat

    # output setups
    output_path = 'data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_file = os.path.join(output_path, 'Xte_hog.npy') 
    np.save(output_file, X_hog)