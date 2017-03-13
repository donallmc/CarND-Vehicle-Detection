import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from feature_extractor import FeatureExtractor
from car_classifier import CarClassifier
from vehicle import Vehicle
from vehicle_fleet import VehicleFleet
from skimage.feature import hog
import time
from scipy.ndimage.measurements import label

class VehicleTracker:
    WINDOW_SIZE = 64
    
    def __init__(self, training_data_path, colour_space, num_orientations,
                     pixels_per_cell, cells_per_block, hog_channel, spatial_size,
                     hist_bins, toggle_spatial_features=True, toggle_histogram_features=True,
                     toggle_hog_features=True):
        
        self.orientations = num_orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        self.feature_extractor = FeatureExtractor(colour_space,
                                                     num_orientations,
                                                     pixels_per_cell,
                                                     cells_per_block,
                                                     hog_channel,
                                                     spatial_size,
                                                     hist_bins,
                                                     toggle_spatial_features,
                                                     toggle_histogram_features,
                                                     toggle_hog_features)
        
        self.classifier = CarClassifier(training_data_path, self.feature_extractor)
        self.fleet = VehicleFleet()
        self.heatmap = None
        self.frames = 0
        self.labels = None

    def convert_colour(self, img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)        

    def passthru(self, img):
        return img

    def crop_image(self, img, boundaries, scale):
        cropped = img[boundaries[2]:boundaries[3],boundaries[0]:boundaries[1],:]             
        if scale != 1:
            imshape = cropped.shape
            cropped =  cv2.resize(cropped, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        return cropped 

    def add_vehicles_to_heatmap(self, img, boundaries, scale):
        if self.heatmap == None:
            self.heatmap = np.zeros_like(img[:,:,0])

        found = 0
        cropped = self.crop_image(img, boundaries, scale)

        xstart = boundaries[0]
        ystart = boundaries[2]

        candidates = self.feature_extractor.generate_windowed_features(cropped, self.WINDOW_SIZE)
        for candidate in candidates:
            scaled = self.classifier.scale_features(candidate[2])
            if self.classifier.is_car(scaled):
                xleft = candidate[0]
                ytop = candidate[1]
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(self.WINDOW_SIZE * scale)
                self.heatmap[ytop_draw + ystart:ytop_draw + win_draw + ystart, xbox_left + xstart:xbox_left + xstart + win_draw ] += 1

    def annotate_image(self, img, scale=1):
        output_img = np.copy(img)
        self.frames += 1        

        #account for jpeg/png scaling difference
        img = img.astype(np.float32)/255

        colour_corrected = self.convert_colour(img, conv='RGB2YCrCb')
        self.add_vehicles_to_heatmap(colour_corrected, (0, 1280, 400, 512), 1)                
        self.add_vehicles_to_heatmap(colour_corrected, (0, 1280, 400, 512), 1.5)                
        self.add_vehicles_to_heatmap(colour_corrected, (0, 1280, 472, 600), 2)
                   
        if self.frames == 20:
            self.labels = label(self.apply_threshold(self.heatmap, 5))
            self.frames = 0
            self.heatmap = np.zeros_like(img[:,:,0])
            
        if self.labels == None:
            self.labels = label(self.apply_threshold(self.heatmap, 2))

        vehicles = self.fleet.process_new_data(self.get_boxes_from_labels(self.labels))
        for vehicle in vehicles:
            cv2.rectangle(output_img, vehicle[0], vehicle[1], (255, 0, 0), 2)
            self.heatmap[vehicle[0][1]:vehicle[1][1],vehicle[0][0]:vehicle[1][0]] = 9

        return output_img

    def apply_threshold(self, heatmap, threshold):
        #cut off the left side of the image as it shouldn't include cars we care about. This will break rather spectacularly if we're not in the leftmost lane :D
        heatmap[:,:700] = 0
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def get_boxes_from_labels(self, labels):
        boxes = []
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            boxes.append(bbox)       

        return boxes
