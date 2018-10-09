# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./imgs/left_2018_10_01_11_05_08_336.jpg "left camera"
[image2]: ./imgs/center_2018_10_01_11_05_08_336.jpg "center camera"
[image3]: ./imgs/right_2018_10_01_11_05_08_336.jpg "right camera"
[image4]: ./imgs/cropped.png "cropped"
[image5]: ./imgs/center_2018_10_02_17_33_05_866.jpg "track2"
[image6]: ./imgs/learning_1.png
[image7]: ./imgs/learning_2.png

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution nueral network for track one
* writeup.md summarizing the results
* video.mp4 simulation result

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I drived one lap in a counter-clockwise.   I used all image of center camera and left camera, right camera of center lane driving. the steering angles are adjusted for the side camera images.

Because of the data is left curve biased, the trained model may not be good for general scene, so I added flipped images to training data sets.

After the collection process, I had 5868 number of data points.

Here are example images of center lane driving(left, center, right):

![alt text][image1]
![alt text][image2]
![alt text][image3]

#### 2. An appropriate model architecture has been employed

My model is based on (nvidia's model)[https://devblogs.nvidia.com/deep-learning-self-driving-cars/], but it has only 4 convolution layers. The model consists of convolution neural network with 5x5 filter and 3x3 filter, and depths between 24 and 64 (model.py lines 88-99). Each convolution layers include RELU activation function and valid padding. Max pooling is used to pool layers.

Fully connected layers consist dence connection with between 2176 neurons and 10 neurons (model.py lines 106-110). 

The data is cropped to see only load lane and is normalized in the model using a Keras Cropping2D layer and a Keras lambda layer (model.py line 85-86).

Here is an example image of cropped data:

![alt text][image4]


My final model consisted of the following layers:
|layer|description|output|
|:-:|:-:|:-:|
|Cropping2D|crop upper 50pix and lower 20pix|90x320x3|
|Lambda|normalize|90x320x3|
|Conv2D|5x5 kernel, 24 chns, relu, valid|86x316x24|
|MaxPooling2D|stride (2,2)|43x158x24|
|Conv2D|5x5 kernel, 36 chns, relu, valid|39x154x36|
|MaxPooling2D|stride (2,2)|19x77x36|
|Conv2D|5x5 kernel, 48 chns, relu, valid|15x73x48|
|MaxPooling2D|stride (2,2)|7x36x48|
|Conv2D|3x3 kernel, 64 chns, relu, valid|5x34x64|
|MaxPooling2D|stride (2,2)|2x17x64|
|Flatten||2176|
|Dence|100 nerrons|100|
|Dence|50 nerrons|50|
|Dence|10 nerrons|10|
|Dence|output|1|


#### 3. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting and shuffle the sets before each epoch (model.py line 132-133). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 5. Training
The number of epochs is 20.

Here is learning curve:

![alt text][image6]

The final valitation loss is 0.0008.

----
### Track two

My project includes the following files for track two:
* model_t2.h5 containing a trained convolution nueral network for track two
* video_track2.mp4 simulation result

#### 1. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Because  track two has two lane where track one has only a lane, instead of flipping images, I used a combination right lane driving in clockwise and right lane driving in counter-clockwise. Both of each driving scenes contain one lap.

Here is an example image:

![alt text][image5]

#### 2. Training

The convolution network model architecture for track two is same as for track one. However the number of epochs is 40.

Here is learning curve:

![alt text][image7]

The final valitation loss is 0.0045.