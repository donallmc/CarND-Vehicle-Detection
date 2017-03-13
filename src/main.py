import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from vehicle_tracker import VehicleTracker
import sys
from moviepy.editor import VideoFileClip
from optparse import OptionParser

parser = OptionParser(usage="usage: %prog [options]")
parser.add_option("-v", "--video_to_process",
                  action="store",
                  dest="video_input",
                  help="a video file to identify vehicles in")
parser.add_option("-o", "--output_filename",
                  action="store",
                  dest="video_output",
                  default="annotated.mp4",
                  help="a target to store the annotated video")
parser.add_option("-i", "--image_to_process",
                  action="store",
                  dest="image_input",
                  help="an image file to identify vehicles in")

(options, args) = parser.parse_args()
if not options.video_input and not options.image_input:
        parser.error('Must specify video file or image file to annotate')
if options.video_input and options.image_input:
        parser.error('Must specify video file or image file; not both')                

if len(args) != 0:
    parser.error("wrong number of arguments")

    
tracker = VehicleTracker(training_data_path="../training_data/",
                             colour_space='YCrCb',
                             num_orientations=12,
                             pixels_per_cell=8,
                             cells_per_block=2,
                             hog_channel= 'ALL',
                             spatial_size=(32,32),
                             hist_bins=64,
                             toggle_spatial_features=True,
                             toggle_histogram_features=True,
                             toggle_hog_features=True)

if options.image_input:
        img = mpimg.imread(options.image_input)
        tracker.annotate_image(img)
elif options.video_input:        
        input_clip = VideoFileClip(options.video_input)
        annotated_clip = input_clip.fl_image(tracker.annotate_image)        
        annotated_clip.write_videofile(options.video_output, audio=False)
