import os
import cv2
import time
import random
import numpy as np
import shutil
from tensorflow.keras.metrics import Precision, Recall

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf



def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (105,105))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

def make_embedding(): 
    inp = Input(shape=(105,105,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()





# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    

l1 = L1Dist()





def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105,105,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(105,105,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()



binary_cross_loss = tf.losses.BinaryCrossentropy()

opt = tf.keras.optimizers.Adam(1e-4) # 0.0001



# # Reload model 
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# img_1 = preprocess('C:\\Users\\omarg\\OneDrive\\Desktop\\FR Project\\Faces\\6\\5.jpg')
# img_2 = preprocess('C:\\Users\\omarg\\OneDrive\\Desktop\\FR Project\\Faces\\6\\8.jpg')

# result = siamese_model.predict(list(np.expand_dims([img_1, img_2], axis=1)))



# print(result[0][0])
def take_attendance(id, img):
    threshHold = 0
    img = preprocess(img)
    rootdir = 'C:\\Users\\omarg\\OneDrive\\Desktop\\FlaskCamera\\students'
    for dir in os.listdir(rootdir):
        if(dir == id):
            d = os.path.join(rootdir, dir)
            for image in os.listdir(d):
                anchor = os.path.join(d, image)
                anchor = preprocess(anchor)
                result = siamese_model.predict(list(np.expand_dims([img, anchor], axis=1)))
                if(result[0][0] > 0.4) :
                    threshHold += 1
            
    if (threshHold >= 0):
        return True
    else:
        return False

# id = "1"

# rootdir = 'C:\\Users\\omarg\\OneDrive\\Desktop\\FlaskCamera\\students'
# for dir in os.listdir(rootdir):
#     print(dir)
#     print(type(dir))
#     print(dir == id )
#     if(dir == id):
#         d = os.path.join(rootdir, dir)
#         for image in os.listdir(d):
#             print(image)
#             anchor = os.path.join(d, image)
#             print(anchor)

# attendance = take_attendance("1",'C:\\Users\\omarg\\OneDrive\\Desktop\\FlaskCamera\\shots\\5ra.png')
# print(attendance)