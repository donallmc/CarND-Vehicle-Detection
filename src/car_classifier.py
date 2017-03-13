import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import time

class CarClassifier:
    VEHICLES_SUBDIR = "vehicles/"
    NON_VEHICLES_SUBDIR = "non-vehicles/"
    
    def __init__(self, training_data_path, feature_extractor):
        cars = self.load_training_imgs(training_data_path + self.VEHICLES_SUBDIR)
        not_cars = self.load_training_imgs(training_data_path + self.NON_VEHICLES_SUBDIR)
        self.feature_extractor = feature_extractor
        self.create_training_data(cars, not_cars)
        self.train()
#        joblib.dump(self.classifier, 'model.pkl') 

    def load_training_imgs(self, training_data_path):
        basedir = training_data_path
        image_types = os.listdir(basedir)
        imgs = []
        for imtype in image_types:
            imgs.extend(glob.glob(basedir + imtype + "/*"))
        return imgs

    def create_training_data(self, cars, not_cars):
#        n_samples = 10
#        random_idxs = np.random.randint(0, len(cars), n_samples)
#        test_cars = np.array(cars)[random_idxs]
#        test_not_cars = np.array(not_cars)[random_idxs]
        test_cars = cars
        test_not_cars = not_cars
        #get rid of this entirelybefore committing

        car_features = self.feature_extractor.extract_features(test_cars)
        not_car_features = self.feature_extractor.extract_features(test_not_cars)


        X = np.vstack([car_features, not_car_features]).astype(np.float64)
        self.feature_scaler = StandardScaler().fit(X)
        X_scaled = self.feature_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=np.random.randint(0, 100))

    def train(self):
#        self.classifier = joblib.load('model.pkl') 
        print('training classifier with', len(self.X_train), 'examples of', len(self.X_train[0]),'features')
        self.classifier = LinearSVC()
        t = time.time()
        self.classifier.fit(self.X_train, self.y_train)
        print(round(time.time()-t, 2), 'seconds to train svc')
        print('test accuracy = ', round(self.classifier.score(self.X_test, self.y_test), 4))

    def is_car(self, scaled_features):
        result = self.classifier.decision_function(scaled_features)
        return result[0] > 1.5

    def scale_features(self, features):
        return self.feature_scaler.transform(features)
        
