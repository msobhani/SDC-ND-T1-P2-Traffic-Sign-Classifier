# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image01]: ./writeup_data/visualization-01.png
[image02]: ./writeup_data/visualization-02.png
[image03]: ./writeup_data/visualization-03.png
[image04]: ./writeup_data/visualization-04.png
[image05]: ./writeup_data/visualization-05.png
[image06]: ./writeup_data/visualization-06.png
[image07]: ./writeup_data/visualization-07.png
[image08]: ./writeup_data/visualization-08.png
[image09]: ./writeup_data/visualization-09.png
[image10]: ./writeup_data/visualization-10.png
[image11]: ./writeup_data/preprocessing.png
[image12]: ./writeup_data/test-images.png
[image13]: ./writeup_data/prediction-result.png
[image14]: ./writeup_data/softmax-01.png
[image15]: ./writeup_data/softmax-02.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

The project submission includes all required files.
* Ipython notebook with code
* HTML output of the code
* A writeup report as markdown

The submission contains the folloiwing contents:
* Dataset summary and exploratory visualization as well as training data distribution per class
* The desctiption of preprocessing techniques used and why these techniques were chosen.
* The details of the architecture, including the type of model used, the number of layers, and the size of each layer. 
* How the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.
* The approach to finding a solution. Accuracy on the validation set is 0.958.
* The test on the new images, their visualization, performance on them, model certainty and softmax probabilities

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/msobhani/SDC-ND-T1-P2-Traffic-Sign-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas as well as numpy libraries to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Each row shows ten different traffic sign samples of one class:

![alt text][image01]
![alt text][image02]
![alt text][image03]
![alt text][image04]
![alt text][image05]
![alt text][image06]
![alt text][image07]
![alt text][image08]
![alt text][image09]

The follow chart demonstrates the the distribution of the training data per class:

![alt text][image10]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In order to generalize the model, some image processing techniques were used. First, the images were converted to grayscale. This was done to get rid of colors of the images, because the shape and geometry of the signs were enough for the model to detect the signs.

Then, the normalization of the iamges was performed. to obtain images with zero mean and equal variance. This was done by adjusting the images with the mean of the images, and then the devision of the result over the standard variation. The preprocessing was done on all training, validation and test datasets. Below is an example of the preprocessing of an colored traffic sign image:

![alt text][image11]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 Grayscale image						| 
| Convolution 5x5		| 1x1 stride, Valid padding, outputs 28x28x6	|
| RELU					|												|
| Dropout				| Keep probability: 0.7							|
| Max pooling			| 2x2 stride, Valid padding, outputs 14x14x6	|
| Convolution 5x5		| 1x1 stride, Valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 stride, Valid padding, outputs 5x5x6		|
| Fully connected		| Output: 400									|
| Dropout				| Keep probability: 0.7							|
| Fully connected		| Output: 120									|
| RELU					|												|
| Fully connected		| Output: 84									|
| RELU					|												|
| Dropout				| Keep probability: 0.6							|
| Fully connected		| Output: 43									|
|						|												|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the same LeNet model presented in the course as the basis. To improve the accuracy of the model, I've added three drop out layers, one after the first convolution, and two after the fully connected layers. I've tried different keep probability values to reach the values presented in the model above. Also I've used ADAM optimizer instead of SGD optimizer to achieve better results.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

َAfter lots of try and error with different parameters, finally I came with the following values:


* Learning Rate = 0.001
* Epoch = 50
* batch Size = 128

Also I used the following keep probabilities:

* Keep probability of dropout at the first convolution layer: 0.7
* Keep probability of dropout at FC 0 layer: 0.6
* Keep probability of dropout  at FC 2 layer: 0.7



My final model results were:
* Training set accuracy of 0,999 (99.9%)
* Validation set accuracy of 0,958 (95.8%)
* Test set accuracy of 0,943 (94.3%)

##### What was the first architecture that was tried and why was it chosen?
The model architecture was based on the LeNet architecture from the training materials.

##### What were some problems with the initial architecture?
There was a need to preprocess the input images, and also because of overfitting, the dropouts were introduced in the model to improve the accuracy of the model.
   
   
##### How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

In order to avoid overfitting of data and improve the accuracy of the model, dropouts were added at the convolution layer as well as the fully connected layers. After some try and error, I finally came up with the numbers mentioned before. The keep probablity of the first drop out at the convolution layer was set to 0.7 to prevent underfitting of data.

##### Which parameters were tuned? How were they adjusted and why?

In addition to the keep probabilities, learning rate, epoch and the batch size were adjusted to obtain the best results. The learning rate was set in a way to achieve the lowest loss in the model, and the fastest training rate possible. The batch size was adjusted to 128, because of computation power limits.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web. The first row are the original iamges, and the second row are them after preprocessing:

![alt text][image12] 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			 							| Prediction								| 
|:-----------------------------------------:|:-----------------------------------------:| 
| Stop										| Stop										| 
| Speed limit (30km/h)						| Speed limit (30km/h)						| 
| Turn right ahead							| Turn right ahead							| 
| End of all speed and passing limits		| End of all speed and passing limits 		|
| Priority road								| Priority road								|
| Vehicles over 3.5 metric tons prohibited	| Vehicles over 3.5 metric tons prohibited	|
| Slippery Road								| Slippery Road 							|
| Wild animals crossing						| Wild animals crossing 					|
        

The model was able to correctly guess all of traffic signs, which gives an accuracy of 100%. Here are the top 4 probabilities for each input traffic sign image:

![alt text][image13] 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Except for "Vehicles over 3.5 metric tons prohibited" which had the prediction probability of 89%, the model was able to predict the rest of the input traffic signs 100% correctly.

| Probability	| Prediction								| 
|:-------------:|:-----------------------------------------:| 
| 100%			| Stop										| 
| 100%			| Speed limit (30km/h)						| 
| 100%			| Turn right ahead							| 
| 100%			| End of all speed and passing limits 		|
| 100%			| Priority road								|
| 89%			| Vehicles over 3.5 metric tons prohibited	|
| 100%			| Slippery Road 							|
| 100%			| Wild animals crossing 					|

The following figure demonstrates the softmax probability for each traffic sign image found on the web:

![alt text][image14] 
![alt text][image15] 

