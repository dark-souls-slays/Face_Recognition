<b>Claudia Espinoza, Asunción Martínez <br>
National Dong Hwa Univerisity <br>
Final Project - Machine Learning </b>

------------------------------------------------------------------------------------------------------------------------------------------
<br>
# Image Classification System
## Problem Introduction:
The task was to build a face recognizer system using a total of 650 images pertaining to 50 different people. Since the data consists of 650 images, 13 images for each different person, it was divided in the following manner:  
Training -> 10 images * 50 different faces
Testing -> 3 images * 50 different faces

To build a convolutional neural network for the images they python package, Keras, was chosen. To tackle this problem in theory, it will be broken down as follows: First, import tha data, feed the training data to the network. After the network has learned to associate images and their corresponding labels, the network will make some predictions on the testing data. The predictions will then be verified and optimized where necessary.

## Solutions
IMPORT DATA <br>
Read directory "training_data" and resize all of the images to (128, 128), given they're all of different sizes. Y contains an array of  500 labels 1-50, used to identify the person in dataset[x]. The reading of data was done through the initial inputs function. 

Once being able to extract the dataset, we form four arrays. Two for the training set - training_set, labels are for the training set and the training set's labels respectively. These are the images the model will learn from. Likewise, we have two for the testing set - test_set, labels_test which are for the testing sets and it's labels respectively; and, these are the images the model will be tested on. <br>
Since images have height, width and color depth, the tensors are formatted as flaot32 of shape (500,128,128,3) for the training data, and (150,128,128,3) for testing. Below is the sample code: <br>

```python
training_set, labels = inputs("training_data")
test_set, labels_test = inputs("test_data")

training_set.reshape(500,128,128,3)
test_set.reshape(150,128,128,3)

#Pixel value normalization
training_set = training_set.astype('float32') / 255.0
test_set = test_set.astype('float32') / 255.0
```
Note that the images are also divided by 255. This is to normalize the data. That is, reshaping and scaling all values for the network into a range between 0 and 1.

LABEL PREPARATION <br>
The labels corresponding to both the training set and test sets are categorically encoded, using one hot encoding. 
```python
labels = to_categorical(labels)
labels_test = to_categorical(labels_test)
```
MODEL <br>
The layers were combined into a network or model. Below we will analyze how our model works. 
NETWORK COMPILATION <br>
TRAINING <br>
## Discussion
