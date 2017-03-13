#**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./examples/car.png
[notcar]: ./images/notcar.png
[hogcar]: ./images/hogcar.png


[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting HOG features is found [here](https://github.com/donallmc/CarND-Vehicle-Detection/blob/master/src/feature_extractor.py#L81-L99). It's based on the code provided in the course materials.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car] ![alt text][notcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][car] ![alt text][hogcar]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and, mostly through trial and error, settled on 9 orientations, 8-pixel cells, and 2x2 cells per block, as well as the YCrCb colour space. Modifying the colour space was the only thing that seemed to produce a statistically significant difference in accuracy. For other parameters I experimented within a fairly narrow range as the accuracy was already high (> 98%) and I was cautious about harming my iteration time by causing the model to take a long time to train.

####2.1 Other features

In addition to the HOG features, I used spatial features and colour histograms as described in the coursework. For the spatial features I settled on a size of 32x32. For colour histograms, I did a bit of tinkering with the number of histograms. I found that relatively large numbers (> 256) didn't seem to have a desirable impact on the accuracy. In the end, I wasn't able to notice a significant difference between 32 and 64 bins, but admittedly there were many other factors involved in the pipeline and more could have been done to isolate the impact of this parameter. Pragmatically speaking, this is the last weekend of the term and it's time to stop coding and submit something! :)

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the aforementioned HOG, spatial, and colour features. The total number of features 10,320 was per exemplar. I used a 90/10 training/test random split on the provided training data (about 16K, evenly split between cars and not-cars) and achieved an accuracy of 98.5%-99.2% (depending on the random split).

The course materials mentioned there were some issues with the data that might cause overfitting and that a manually-curated train/test split could improve matters. I didn't get as far as doing that, but I believe my model's performance in processing the video was negatively affected by it!

I attempted to add some additional car images to increase the test set. I considered the usual manipulations to augment it, but cropping and translation seemed like they might have a bad effect. flipping vertically would obviously be bad but horizontally might have been useful, although I didn't have time to evaluate. Modifying brightness would also probably have been a useful thing to do.

Based on some comments on the forums I attempted to use a standard sklearn SVC classifier with a linear kernel but the training time was very long so I abandoned it. The aim in using SVC was to get a better probability output from the classifier when predicting a label for an image. Instead of doing that, I implemented a threshold for the distance from the hyperplane provided by the LinearSVC. Through trial and error, a distance of 1.5 seemed to work well, although this definitely erred on the side of fewer false positives in the video.

The classifier and all related operations are encapsulated in [this class](https://github.com/donallmc/CarND-Vehicle-Detection/blob/master/src/car_classifier.py).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding windows is taken from the video that Ryan did. I modified it slightly to fit my class structure and it can be seen [here](https://github.com/donallmc/CarND-Vehicle-Detection/blob/master/src/feature_extractor.py#L101-L141). That code is invoked by a [function](https://github.com/donallmc/CarND-Vehicle-Detection/blob/master/src/feature_extractor.py#L101-L141) that applies "hits" to the heatmap. That function, in turn, is invoked 3 times [here](https://github.com/donallmc/CarND-Vehicle-Detection/blob/master/src/vehicle_tracker.py#L91-L93). Each time the function is invoked represents a different size window operating in a different area. I initially restricted both the x and y axes but subsequently allowed the entire range of x values. This could definitely be improved to speed up processing time, ideally in concert with a road-finding module that dynamically decided where to look.

The Y lanes chosen for each window size (64px, 96px, 128px) were chosen by trying different boundaries and examining the resulting heat maps. These values were chosen as a balance between minimizing both false positives and processing time. The 3 chosen sizes were obviously based on being multiples of the cell sizes. Windows larger or smaller than these were found not to produce very good results.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

In addition to the heatmap, I implemented [a class](https://github.com/donallmc/CarND-Vehicle-Detection/blob/master/src/vehicle_fleet.py) to track the progress of individual vehicles in the video. This class assumes that vehicles don't vanish in the middle of the road and that they enter/exit the field of vision from a specific area of an image (in the case of the video, cars enter to the right of the camera position). The class I provide is simple and tailored for the video but could be expanded to be more universal.

A clear advantage of tracking vehicles this way is that it can eliminate a lot of spurious false positives by simply ignoring them if they pop up in certain regions of the image where we don't expect to see vehicles or if they pop up in an area where vehicles can be found but where we haven't observed any vehicles yet. This approach also enabled the video to fill in frames where a vehicle was temporarily lost by retaining the vehicle for some time after it disappeared (while in the centre of the image) and by projecting where the vehicle might reappear based on the trajectory traveled before it disappeared.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue with this whole project is the accuracy of the classifier. Although the accuracy was high with the training/test set, the number of false positives was significantly higher than expected and several car images were not classified correctly, implying a significant degree of overfitting. Were this to be a production system, I would begin by gathering a lot more training data and designing a much more robust model.

The heatmap approach is a very interesting method for identifying vehicle location, but ideally it would be only one feature of many used to train a vehicle detector, including sensor data, etc.

As briefly touched upon in the previous section, a significant failure point in my project is the vehicle tracker. It was coded for _this_ project, which means it makes certain assumptions, like the position of the car being in the left-most lane. A more robust solution would use a road-detection module to map the road out and decide where to look for vehicles and where they can enter/exit (not just tp/from either side of the vehicle and the horizon, but also at intersections, etc).

Additionally, the vehicle tracking is based on heuristics for when to assume a vehicle is still there even though we can't see it, when to assume it has disappeared, etc. This is a clear candidate for a more versatile, machine-learned solution.
