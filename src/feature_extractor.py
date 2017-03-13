from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

#a lot of this file is heavily based on coursework and the youtube video in which Ryan walked through an implementation
class FeatureExtractor:

    def __init__(self, colour_space,
                     orientations,
                     pixels_per_cell,
                     cells_per_block,
                     hog_channel,
                     spatial_size,
                     hist_bins,
                     toggle_spatial_features=True,
                     toggle_histogram_features=True,
                     toggle_hog_features=True):
        self.toggle_hog_features = toggle_hog_features
        self.toggle_histogram_features = toggle_histogram_features
        self.toggle_spatial_features = toggle_spatial_features
        self.hist_bins = hist_bins
        self.spatial_size = spatial_size
        self.hog_channel = hog_channel
        self.cells_per_block = cells_per_block
        self.pixels_per_cell = pixels_per_cell
        self.orientations = orientations
        self.colour_space = colour_space
        print('Using: ', self.orientations, 'orientations,', self.pixels_per_cell, 'pixels per cell,', self.cells_per_block, 'cells per block,', self.hist_bins, 'histogram bins,',self.spatial_size,'spatial sampling',self.colour_space,'colour space')

    def extract_features(self, imgs):
        features = []
        for img in imgs:
            img_features = []
            img = mpimg.imread(img)
            if(img.shape != (64, 64, 3)):
                print("resizing")
                img = cv2.resize(img, (64, 64))
            colour_corrected = self.transform_colour_space(img)
            if self.toggle_spatial_features:
                self.spatial_features = self.gen_spatial_features(img)
                img_features.append(self.spatial_features)
            if self.toggle_histogram_features:
                self.histogram_features = self.gen_histogram_features(img)
                img_features.append(self.histogram_features)
            if self.toggle_hog_features:
                self.hog_features = self.gen_hog_features(img)
                img_features.append(self.hog_features)
            features.append(np.concatenate(img_features))
        return features
                
    def transform_colour_space(self, img):
        if self.colour_space != 'RGB':
            if self.colour_space == 'HSV':
                corrected = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.colour_space == 'LUV':
                corrected = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.colour_space == 'HLS':
                corrected = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.colour_space == 'YUV':
                corrected = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.colour_space == 'YCrCb':
                corrected = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            corrected = np.copy(img)
        return corrected

    def gen_spatial_features(self, img):
        ch1 = cv2.resize(img[:,:,0], self.spatial_size).ravel()
        ch2 = cv2.resize(img[:,:,1], self.spatial_size).ravel()
        ch3 = cv2.resize(img[:,:,2], self.spatial_size).ravel()
        return np.hstack((ch1, ch2, ch3))

    def gen_histogram_features(self, img):
        ch1_hist = np.histogram(img[:,:,0], bins=self.hist_bins)
        ch2_hist = np.histogram(img[:,:,1], bins=self.hist_bins)
        ch3_hist = np.histogram(img[:,:,2], bins=self.hist_bins)
        return np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))

    def gen_hog_features(self, img):
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(self.get_hog_features_per_channel(img[:,:,channel]))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = self.get_hog_features_per_channel(img[:,:,self.hog_channel])

        return hog_features
    
    def get_hog_features_per_channel(self, img_channel, vector=True):
        return hog(img_channel,
                       orientations=self.orientations,
                       pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                       cells_per_block=(self.cells_per_block, self.cells_per_block),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=vector)

    def generate_windowed_features(self, img, window=64):
        windowed_features = []
        ch1 =img[:,:,0]
        ch2 = img[:,:,1]
        ch3 = img[:,:,2]        

        nxblocks = (ch1.shape[1] // self.pixels_per_cell) - 1
        nyblocks = (ch1.shape[0] // self.pixels_per_cell) - 1            

        nfeat_per_block = self.orientations * self.cells_per_block ** 2
        nblocks_per_window = (window // self.pixels_per_cell) - 1
        cells_per_step = 2
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step            

        hog1 = self.get_hog_features_per_channel(ch1, vector=False)
        hog2 = self.get_hog_features_per_channel(ch2, vector=False)
        hog3 = self.get_hog_features_per_channel(ch3, vector=False)        

        count = 0
        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()                
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).astype(np.float64)

                xleft = xpos * self.pixels_per_cell
                ytop = ypos * self.pixels_per_cell

                subimg = cv2.resize(img[ytop:ytop+window, xleft: xleft+window], (64, 64))

                spatial_features = self.gen_spatial_features(subimg)
                hist_features = self.gen_histogram_features(subimg)
                windowed_features.append((xleft, ytop, np.hstack((spatial_features, hist_features, hog_features)).astype(np.float64)))
        return windowed_features
