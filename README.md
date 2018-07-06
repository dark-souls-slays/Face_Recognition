<b>Claudia Espinoza, Asunción Martínez <br>
National Dong Hwa Univerisity <br>
Final Project - Machine Learning </b>

------------------------------------------------------------------------------------------------------------------------------------------
<br>

## Image Classification System - Problem Introduction 
The task was to build a face recognizer system using a total of 650 images pertaining to 50 different people. Since the data consists of 650 images, 13 images for each different person, it was divided in the following manner:  <br>
Training -> 10 images * 50 different faces <br>
Testing -> 3 images * 50 different faces <br>

To build a convolutional neural network for the images the python package, Keras, was chosen. To tackle this problem in theory, it will be broken down as follows: First, import tha data, feed the training data to the network. After the network has learned to associate images and their corresponding labels, the network will make some predictions on the testing data. The predictions will then be verified and optimized where necessary.

## Solutions
<b>IMPORT DATA</b> <br>
Read directory "training_data" and resize all of the images to (128, 128), given they're all of different sizes. Y contains an array of  500 labels 1-50, used to identify the person in dataset[x]. The reading of data was done through the initial inputs function. 

Once being able to extract the dataset, we form four arrays. Two arrays are for the training set. The variable names: <i>training_set, labels</i> are for the training set and the training set's labels respectively. These are the images the model will learn from. Likewise, we have two for the testing set - test_set, labels_test which are for the testing sets and it's labels respectively; and, these are the images the model will be tested on. <br>
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

<b>LABEL PREPARATION </b><br>
The labels corresponding to both the training set and test sets are categorically encoded, using one hot encoding. 
```python
labels = to_categorical(labels)
labels_test = to_categorical(labels_test)
```
<b>MODEL </b><br>
Next, layers were combined into a network or model. The sequential function shows the initializing of our layers. Below we will analyze how our model works. <br>
Firstly, we used convolutional layers to learn <i>local patterns</i> that are found in 2D windows of the inputs. Given convolutions operate over feature maps, it extracts patches from the input and produces an output feature map. The output depth is a parameter of the layer, and its channels can be thought of as filters. Thus, the output depth can be passed onto the Conv2D layer as arguments. We start with a depth of 32 in our first layer with 'relu' activation and an input of images of 128x128 in color. <br>
Second, because convolutions are being done in 3x3 windows(as per parameters chosen) Maxpooling2D layer is added. The reason behind is that although the max pooling extracts windows from input feature maps, and output the max value of each channel (similar to the Conv2D), they are usually done in 2x2 windows. Hence, they can significantly downsample feature maps. And thus, this reduces the number of feature map cooefecients that will be processed. Furthermore, max pooling can make the next convolutional layer look at larger windows, which is essentially what we did here. <br>
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
```
After going through 3 different convolutions each of increasing depths, 32, 64, and 128, we reach the Flatten layer. A dropout was immediately inserted for regularization. This dropout serves to randomly dropping out a number of output features of the layer during training; and consequently, dropout helps to reduce overfitting. <br>
Dense layers were now introduced to learn <i>global patterns</i> involving all pixels of the images being inputted. This network consists of a chain of two Dense layers, and each layer is responsible for applying operations on the weight tensors. 

<b>NETWORK COMPILATION </b><br>
In order to have feedback on the learning of the weight tensors, a loss function was used. We used binary_crossentropy for the loss because we are working on a binary classification problem, with parameters customized on the RMSprop of the loss funtion. The accuracy of the images that were correctly classified will be metrics we are interested in monitoring during our training and testing. Therefore, the following was set:
```python
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```
<b>TRAINING </b><br>
The model was now able to start iterating on the training data in mini-batches of 5. The iterations were done 15 times over, or over 15 epochs. On each epoch, the model computed weight gradients as per the loss on the batch, and updated the weights.
```python
model.fit(training_set,labels, batch_size=5, epochs=15)
```
## Results and Discussion
<b>Results </b>
Accuracy: <br>
Loss: <br>
Validation: <br>

<b>Discussion </b><br>
While doing this project, the main obstacle encountered was the fact that the datasets to learn from were extremely small. Due to this, it makes it easy to run into problems, such as overfitting, in the implementation. To fight this it was important that we downsample the features before running any other layers. In this way, the upcoming layer with higher depth didn't become costly or too large to deal with. Initially, our layers were set to lower depths because we were testing how small or large windows would affect the accuracy. It was understood, that with downsampling it is easier for the following layers to look at larger windows, which in turn started to show higher accuracy. Furthermore, dropping out features, tuning parameters, and using regularization techniques before our Dense layers significantly improved the learning of global patterns in our model. <br><br>

Despite the fact that an accuracy in the 90's was achieved, the team members learned that there were still images that were not correctly classified. As previously mentioned, this is a problem often fought in learning small datasets because overfitting becomes more prevalent. As a result the network tends to perform a little poorer on data that it hasn't seen before. <b>To improve this<b>, we could have resorted to applying other deep learning methods such as data augmentation, and/or using a pretrained model. <br><br>
  
<b>Responsibilities Shared </b><br>
One team member firstly implemented the network while the other observed and learned the discussed ideas being implented on code. After having reached a certain accuracy, the second team member went in to proofread the code and find ways to improve the accuracy. In addition, since the second team member also had an understanding of the code, this team member became responsible of writing and explaining the methods implemented here. 
