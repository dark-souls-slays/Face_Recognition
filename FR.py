import numpy as np
from PIL import Image
import glob
import os, os.path
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.utils import to_categorical
from keras.optimizers import RMSprop

def inputs(ID):
    imgs = []
    dataset = []
    y = []
    check = 0
    size = 128, 128
    path = "/Users/ClaudiaEspinoza/Desktop/Face Recognition System/"+ str(ID)
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

trainingset, labels = inputs("training_data")
testset, labelsTest = inputs("test_data")
epochs = 20

#Pixel value normalization
trainingset = trainingset.astype('float32')
testset = testset.astype('float32')
trainingset /= 255.0
testset /= 255.0

# one hot encoding
y_train = to_categorical(labels)
y_test = to_categorical(labelsTest)
# invert encoding
#inverted = argmax(encoded[0])
#print(inverted)

trainingset.reshape(500,128,128,3)
testset.reshape(150,128,128,3)
print(trainingset)
input_shape = (128,128,3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(51, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(trainingset,y_train, batch_size=5, epochs=15)

#Testing
scores = model.evaluate(testset, y_test)
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])


#Predict
#for i in range (10):
    #m = Image.fromarray(testset[499].reshape(128,128), 'RGBA')
    #m.show()
count = 0
prediction = model.predict_classes(testset)
print("Predicting...")
for i in range(150):
    print("Predicting...")
    print(prediction[i])
    print(np.argmax(y_test[i]))
    if prediction[i] == np.argmax(y_test[i]):
        count = count + 1
print(float(count)/150.0)
