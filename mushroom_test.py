#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


os.environ['TFHUB_CACHE_DIR'] = '/Users/leejiwon/Downloads' #Any folder that you can access
elmo_model = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/4', trainable=True)


# In[ ]:


def show_image(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(228, 228))
    plt.imshow(img)
    plt.show()


# In[ ]:


def preprocess_image(item):
    image_string = tf.io.read_file(item[0])
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, (IMAGE_SIZE, IMAGE_SIZE))
    image_resized = tf.cast(image_resized, tf.float32) / 255.0
    return image_resized, tf.strings.to_number(item[1], tf.int64)


# In[ ]:


def get_dataset(edible_fungies, poisonous_fungies, mode, batch_size):
    x = list(edible_fungies) + list(poisonous_fungies)
    y = [0] * len(edible_fungies) + [1] * len(poisonous_fungies)
    items = [(a, b) for (a, b) in zip(x, y)]
    dataset = tf.data.Dataset.from_tensor_slices(np.array(items)).shuffle(len(x))
    dataset = dataset.map(preprocess_image).batch(batch_size)
    return dataset


# In[ ]:


def get_balanced_dataset(edible_fungies, poisonous_fungies, batch_count, batch_size, mode="train"):
    length_per_category = batch_size * batch_count // 2
    edible_indices = np.random.choice(len(edible_fungies), length_per_category)
    poisonous_indices = np.random.choice(len(poisonous_fungies), length_per_category)
    samle_count = 2 * length_per_category
    return get_dataset(
        edible_fungies[edible_indices], 
        poisonous_fungies[poisonous_indices], 
        mode, 
        batch_size
    ), samle_count


# In[ ]:


base_path = "/Users/leejiwon/Downloads/archive/"
lables = ["edible", "poisonous"]
directory_group = [
    ['edible mushroom sporocarp', 'edible sporocarp'], 
    ['poisonous mushroom sporocarp', 'poisonous sporocarp']
]
edible_fungies = []
poisonous_fungies = []
for (label, directories) in zip(lables, directory_group):
    for directory in directories:
        items = os.listdir(base_path + directory)
        for item in items:
            file_path = base_path + directory + "/" + item
            if label == "edible":
                edible_fungies.append(file_path)
            else:
                poisonous_fungies.append(file_path)
edible_fungies = list(set(edible_fungies))
poisonous_fungies = list(set(poisonous_fungies))


# In[ ]:


batch_size = 32
validation_split = 0.2
edible_fungies_split_index = int((1 - validation_split) * len(edible_fungies))
poisonous_fungies_split_index = int((1 - validation_split) * len(poisonous_fungies))
train_edible_fungies, valid_edible_fungies = edible_fungies[:edible_fungies_split_index],  edible_fungies[edible_fungies_split_index:] 
train_poisonous_fungies, valid_poisonous_fungies = poisonous_fungies[:poisonous_fungies_split_index],  poisonous_fungies[poisonous_fungies_split_index:] 
print(len(train_edible_fungies), len(valid_edible_fungies))
print(len(train_poisonous_fungies), len(valid_poisonous_fungies))
num_batch_per_epoch = min(len(train_edible_fungies), len(train_poisonous_fungies)) // batch_size
print(num_batch_per_epoch)
num_epochs = 50
train_edible_fungies = np.array(train_edible_fungies)
valid_edible_fungies = np.array(valid_edible_fungies)
train_poisonous_fungies = np.array(train_poisonous_fungies)
valid_poisonous_fungies = np.array(valid_poisonous_fungies)
total_valid_count = len(valid_edible_fungies) + len(valid_poisonous_fungies)


# In[ ]:


len(edible_fungies)


# In[ ]:


len(poisonous_fungies)


# In[ ]:


edible_fungies[:10]


# In[ ]:


poisonous_fungies[:10]


# In[ ]:


for i in range(10):
    show_image(poisonous_fungies[np.random.randint(len(poisonous_fungies))])


# In[ ]:


for i in range(10):
    show_image(edible_fungies[np.random.randint(len(edible_fungies))])


# In[35]:


IMAGE_SIZE = 224
handle_base = "mobilenet_v2"
MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
feature_extractor.trainable = False


# In[36]:


tf.keras.backend.clear_session()
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.summary()


# In[37]:


valid_dataset = get_dataset(valid_edible_fungies, valid_poisonous_fungies, "valid", batch_size)
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy()
history = {
    "train_loss": [],
    "valid_loss": [],
    "train_accuracy": [],
    "valid_accuracy": []
}
for epoch in range(num_epochs):
    begin_time = time.time()
    train_dataset, total_train_count = get_balanced_dataset(train_edible_fungies, train_poisonous_fungies, num_batch_per_epoch, batch_size, mode="train")
    train_losses = []
    valid_losses = []
    correct_count = 0
    total_count = 0
    for (x_batch, y_true) in train_dataset:
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            predict_labels = tf.argmax(y_pred, axis=-1)
            loss_value = loss(y_true, y_pred)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_losses.append(loss_value)
        correct_count += tf.reduce_sum(tf.cast(y_true == predict_labels, tf.int64))
        total_count += y_true.shape[0]
    train_loss = tf.reduce_mean(train_losses)
    train_accuracy = correct_count / total_train_count
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    correct_count = 0
    total_count = 0
    for (x_batch, y_true) in valid_dataset:
        y_pred = model(x_batch)
        predict_labels = tf.argmax(y_pred, axis=-1)
        loss_value = loss(y_true, y_pred)
        valid_losses.append(loss_value)
        correct_count += tf.reduce_sum(tf.cast(y_true == predict_labels, tf.int64))
        total_count += y_true.shape[0]
    valid_loss = tf.reduce_mean(valid_losses)
    valid_accuracy = correct_count / total_valid_count
    history["valid_loss"].append(valid_loss)
    history["valid_accuracy"].append(valid_accuracy)
    elapsed_time = time.time() -  begin_time
    print("Epoch: %d / %d"%(epoch + 1, num_epochs))
    print("%.2fs Loss: %.2f Accuracy: %.2f Validation Loss: %.2f Validation Accuracy: %.2f"%(elapsed_time, train_loss, train_accuracy, valid_loss, valid_accuracy))
for key in history:
    history[key] = list(np.array(history[key]))


# In[40]:


pd.DataFrame(history).plot()


# In[41]:


predicted_labels = []
actual_labels = []
for (x_batch, y_true) in valid_dataset:
    y_pred = model(x_batch)
    predicted_labels += list(np.array(tf.argmax(y_pred, axis=-1)))
    actual_labels += list(np.array(y_true))


# In[42]:


matrix = confusion_matrix(actual_labels, predicted_labels)
print(matrix)
sns.heatmap(matrix)


# In[49]:


cls_report = classification_report(predicted_labels, actual_labels)
print(cls_report)


# In[56]:


from tensorflow.keras.models import load_model


# In[58]:


model.save("mushroom_model.h5")


# In[61]:


new_model = tf.keras.models.load_model('mushroom_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
new_model.summary()


# In[ ]:




