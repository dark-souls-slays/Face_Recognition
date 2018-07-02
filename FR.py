"""
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
"""
import numpy as np
from PIL import Image
import glob
import os, os.path
import tensorflow as tf
import matplotlib.pyplot as plt

#Load 650 images as (261, 161, 3) -> (261,161) dimension
def inputs():
    imgs = []
    dataset = []
    path = "/Users/ClaudiaEspinoza/Desktop/Face Recognition System/Face Database"
    for f in os.listdir(path):
        im = Image.open(os.path.join(path,f))
        im = im.convert('1') # convert image to black and white
        #im.save("r.jpg")
        im.load()
        data = np.array(im)
        data = data.astype(int)
        im.close()
        dataset.append(data)
    dataset = np.array(dataset)
    print(dataset[0])
    return dataset


dataset = inputs()
"""
from scipy.misc import face
img = face(gray=True)
print(img.shape)
plt.imshow(img, cmap=plt.cm.gist_gray)
plt.show()
"""
