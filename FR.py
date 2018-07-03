import numpy as np
from PIL import Image
import glob
import os, os.path
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras import optimizers

def inputs(ID):
    imgs = []
    dataset = []
    y = []
    check = 0
    size = 128, 128
    path = "/home/asuncion/Documents/ML/Face_Recognition/"+ str(ID)
    for f in os.listdir(path):
        if f == '.DS_Store':
            continue
        im = Image.open(os.path.join(path,f))
        #im = im.convert('L')
        im = im.resize((size), Image.ANTIALIAS)
        #im.load()
        data = np.array(im)
        im.close()
        #data = np.reshape(data,16384)
        dataset.append(data)
        y.append(int(f[1:3]))

    dataset = np.array(dataset)
    y = np.array(y)
    #CHECK THAT THIS PARTICULAR IMAGE CORRESPONDS WITH ITS Y LABEL
    """
    m = Image.fromarray(dataset[499].reshape(128,128), 'L')
    m.save("this.jpg")
    print(y[499])
    """
    return (dataset,y)

training_set, labels = inputs("training_data")
test_set, labels_test = inputs("test_data")
epochs = 20

#Pixel value normalization
training_set = training_set.astype('float32') / 255.0
test_set = test_set.astype('float32') / 255.0

# one hot encoding
labels = to_categorical(labels)
labels_test = to_categorical(labels_test)
# invert encoding
#inverted = argmax(encoded[0])
#print(inverted)

training_set.reshape(500,128,128,3)
test_set.reshape(150,128,128,3)
print(training_set)
input_shape = (128,128,3)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(51, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(training_set,labels, batch_size=5, epochs=15)

#Testing
scores = model.evaluate(test_set, labels_test)
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])


#Predict
#for i in range (10):
    #m = Image.fromarray(testset[499].reshape(128,128), 'RGBA')
    #m.show()
count = 0
prediction = model.predict_classes(test_set)
print("Predicting...")
for i in range(150):
    print("Predicting...")
    print(prediction[i])
    print(np.argmax(labels_test[i]))
    if prediction[i] == np.argmax(labels_test[i]):
        count = count + 1
print(float(count)/150.0)
