import numpy as np
from PIL import Image
import glob
import os, os.path
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
#Load 650 images as (261, 161, 3) -> (261,161) dimension
def inputs():
    imgs = []
    dataset = []
    y = []
    check = 0
    size = 128, 128
    path = "/Users/ClaudiaEspinoza/Desktop/Face Recognition System/Face Database"
    for f in os.listdir(path):
        im = Image.open(os.path.join(path,f))
        im = im.convert('L')
        im = im.resize((size), Image.ANTIALIAS)
        #im.load()
        data = np.array(im)
        im.close()
        data = np.reshape(data,16384)
        dataset.append(data)
        y.append(int(f[1:3]))
        print("image" + f[1:3])
        print(data)

    dataset = np.array(dataset)
    #CHECK THAT THIS PARTICULAR  IMAGE CORRESPONDS WITH ITS Y LABEL
    m = Image.fromarray(dataset[649].reshape(128,128), 'L')
    m.save("this.jpg")
    print(y[649])
    return (dataset,y)
"""
def classifier():
    samples = 1000
    features = 3
    x, y = make_classification(n_samples=samples, n_features=features, n_redundant=0,
	n_classes=50, n_clusters_per_class=3, n_informative=3)
    print(x.shape)
    print(y.shape)
"""
dataset = inputs()
#classifier()
#softmax used to get the probabilities of each class
