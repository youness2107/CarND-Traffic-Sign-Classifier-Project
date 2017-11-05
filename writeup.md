#**Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./writeup-images/NumTestImgPerClass.png "Visualization"
[image2]: ./writeup-images/NumTrainImgPerClass.png "Grayscaling"
[image3]: ./writeup-images/NumTrainImgPerClass.png "Random Noise"
[keep_right]: ./traffic-images-online/keep_right.png "Keep Right"
[left_arrow]: ./traffic-images-online/left_arrow.png "Left Arrow"
[no_entry]: ./traffic-images-online/no_entry.png "No Entry"
[speed_30]: ./traffic-images-online/speed_30.png "Speed 30"
[speed_50]: ./traffic-images-online/speed_50.png "Speed 50"
[stop_angle]: ./traffic-images-online/stop_angle.png "Stop Sign"
[stop_sign]: ./traffic-images-online/stop_sign.png "Stop Sign"
[straight]: ./traffic-images-online/straight.png "Straight"

## Rubric Points
###Data Set Summary & Exploration

####1. Data Set Summary

I used the pickled dataset for convenience.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set:

![alt text][image1]

![alt text][image2]

![alt text][image3]



###Design and Test a Model Architecture

####1. Process

The first step was to convert the images to greyscale by simply averaging R, G and B values.

I normalized the image using the simple technique described in the class X-128/128.

I did not generate additional data because I wanted to evaluate how much I can get out of the architecture of the network without feeding it any extra data.


####2. Description of the final model

My final model consisted of the following layers:


Input: 32x32x1 grayscale image

-------------------------- Layer 1 -------------------------------------------

Convolution = 5x5 kernel with a stride of 1, Valid Padding, Output 28x28x6

Relu = Activation function chosen is Relu 

Max Pooling = 2x2 kernel size with a stride of 2, Valid Padding Output 14x14x6

-------------------------- Layer 2 --------------------------------------------

Convolution = 5x5 kernel with a stride of 1, Valid Padding, Output 10x10x16

Relu = Activation function chosen is Relu 

Max Pooling = 2x2 kernel size with a stride of 2, Valid Padding Output 5x5x16

-------------------------- Layer 3 -------------------------------------------

Convolution = 3x3 kernel with a stride of 1, Valid Padding, Output 3x3x400

Relu = Activation function chosen is Sigmoid 

Max Pooling = 2x2 kernel size with a stride of 2, Valid Padding, Output 1x1x400

----------------------- Flatten Then Dropout Layer ---------------------------------

Flatten both Layer 2 and Layer 3 and concatenate them 
(5x5x16 = 400 neurons from layer 2 and 1x1x400 = 400 neurons from layer 3)
gives 800 neurons

Dropout 

---------------------- Layer 4 ------------------------

Fully Connected layer of 800 to 43 neurons


####3. Model training notes

To train the model, I used an AdamOptimizer (same as the LeNet Lab).

The Adam Optimizer is a good enhancement over the basic Gradient Descent one.

I used a batch size of 1024 (I have a beefy GPU so I wanted to experiment with bigger batch sizes)

The number of epochs that I used is 1200. This might seem a bit exaggerated but I want to see how much I can squeeze out of the network by training it for a good amount of time. 

I left the hyperparameter mu and sigma the same as the original LeNet as they were giving a good results.

####4. Description of the approach


First Architecture tried was LeNet because it seems to be appropriate for the task and suggested in the class

The initial architecture did not have enough layers to adapt to the increased number of classes in the traffic dataset compared to MNIST. The initial accuracy was not that impressive (around 90%)

I adjusted LeNet architecture to be closer to the one described on the paper suggested in the project
([link to paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf))

Also I added a dropout layer to make it possible to train the network for a long amount of time (1200 epochs) without seeing any drop on the validation accuracy (on average per 10 epochs)

The learning rate was tuned by doing a binary search from 0.0005 to 0.001. 
0.008 seemed to be working pretty good for me.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.5% 
* test set accuracy of 95.8%
 

###Test a Model on New Images

####1. Testing with new images from the web

Here are 7 German traffic signs that I found on the web: (most of them are screenshots using Google Street View)

![alt text][keep_right]
![alt text][left_arrow]
![alt text][no_entry]
![alt text][speed_30]
![alt text][stop_angle]
![alt text][stop_sign]
![alt text][straight]

The quality of the images is not bad and I believe the model shouldn't have any problem classifying them


####2. Results on the new images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| Turn left ahead    	| Turn left ahead  								|
| No entry				| No entry										|
| Speed limit (30km/h)	| Speed limit (30km/h)					 		|
| Stop  1  				| Stop 											|
| Stop	2				| Stop											|
| Ahead only			| Ahead only					 				|


The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%. 

This is a perfect accuracy (around 5% more than the test set) but still not realistic as only 7 images were considered.

####3. Softmax Probabilities on the new images

For all images the probability is almost 1 and the model is sure about the classification.
(Most likely this is due to the very high number of epochs used in the training)

Below few of the results with the probabilities of the 5 top candidates
(No need to list them for all the images as they were all very close to 1)

First Image : Keep Right Sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00     	| Keep right 									| 
| 3.65709440e-15    	| Slippery road									|
| 6.34509734e-16		| Priority road									|
| 5.07030587e-16	   	| Speed limit (50km/h)					 		|
| 2.19745440e-16	    | General caution     							|

Second Image: Turn left ahead  Sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00     	| Turn left ahead								| 
| 1.84884446e-08    	| Keep right									|
| 1.14551302e-08		| Right-of-way at the next intersection			|
| 5.85653304e-10	   	| Ahead only				 					|
| 2.95157215e-12	    | Priority road     							|

Third Image: No entry

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00     	| No entry										| 
| 5.30275625e-14    	| Stop											|
| 4.20760202e-16		| No passing									|
| 3.07622368e-23 	   	| Bicycles crossing			 					|
| 1.58027583e-23	    | Speed limit (60km/h)   						|

Fourth Image: Speed limit (30km/h)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00     	| Speed limit (30km/h)							| 
| 5.35239575e-10    	| End of speed limit (80km/h)					|
| 8.21898353e-20		| Speed limit (80km/h)							|
| 6.34561348e-21 	   	| Speed limit (100km/h)		 					|
| 3.51831136e-21	    | Roundabout mandatory  						|

Fifth Image: Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99890447e-01     	| Stop											| 
| 1.08746455e-04    	| Yield											|
| 4.53487814e-07		| Turn Left Ahead								|
| 2.09722941e-07 	   	| Keep Right		 					    	|
| 6.70355647e-08	    | Speed limit (70km/h)  						|






