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
Next, layers were combined into a network or model. The sequential function shows the initializing of our layers. Below we will analyze how our model works. <br>
Firstly, we used convolutional layers to learn local patterns that are found in 2D windows of the inputs. Given convolutions operate over feature maps, its extracts patches from the input and produces and output feature map. The output depth is a parameter of the layer, and its channels can be thought of as filters. Thus, the output depth can be passed onto the Conv2D layer as arguments. We start with a depth of 32 in our first layer with 'relu' activation and an input of images of 128x128 in color. <br>
Second, because convolutions are being done in 3x3 windows(as per parameters chosen) Maxpooling2D layer is added. The reason behind is that although the max pooling extracts windows from input feature maps, and output the max value of each channel, they are usually done in 2x2 windows. Hence, they can significantly downsample feature maps. And thus, this reduces the number of feature map cooefecients that will be processed. Furthermore, max pooling can make the next convolutional layer look at larger windows, which is essentially what we did here. <br>
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
```
After going through 3 different convolutions each of increasing depths, 32, 64, and 128, we reach the Flatten layer. A dropout is now inserted for regularization. This dropout serves to randomly dropping out a number of output features of the layer during training; and consequently, dropout helps to reduce overfitting. <br>
Dense layers are now introduced to learn global patterns involving all pixels of the images being inputted. 

NETWORK COMPILATION <br>
TRAINING <br>
## Discussion
