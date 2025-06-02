#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub 
import matplotlib.pyplot as plt
import sklearn
import time
import pandas as pd
import keras
from sklearn.metrics import classification_report, confusion_matrix


# In[7]:


IMAGE_SIZE = 224
handle_base = "mobilenet_v2"
MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
feature_extractor.trainable = False


# In[8]:


from tensorflow.keras.models import load_model
new_model = tf.keras.models.load_model('mushroom_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
new_model.summary()


# In[19]:


import cv2
im = cv2.imread('test_2.jpg') #사진 읽어들이기

#이미지 출력
import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()


# In[20]:


im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #색공간 변환
im = cv2.resize(im, (224,224)) #사이즈 조정

#이미지 출력
import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()


# In[18]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

model = new_model()

image = load_img('test_2.jpg', target_size=(224, 224))

model.predict(image)


# In[ ]:




