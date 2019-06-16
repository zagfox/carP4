# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

At high level, the model is LeNet architecture with two convolution layers followed by a fully connected layer.

#### 2. Attempts to reduce overfitting in the model

Each convolutional layer has max pooling layer to limit overfitting.

A drop out layer after the second convolution layer is added.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Default training data is used.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model is borrowed from class video. A LeNet with only one fully connected layer.

#### 2. Final Model Architecture

Layer 1: Cropping and nomalizing.
Layer 2: Convolution, relu, activation
Layer 3: Convolution, relu, activation
Layer 4: Dropout
Layer 5: Fully connected
